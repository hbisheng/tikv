// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.

// #[PerformanceCriticalPath]
use concurrency_manager::MaxTsUpdateSource;
use protobuf::Message;
use resource_metering::record_network_out_bytes;
use tikv_util::Either;
use txn_types::{Key, Lock, WriteType};

use crate::storage::{
    ProcessResult, Snapshot,
    kv::WriteData,
    lock_manager::LockManager,
    mvcc::{MvccTxn, OverlappedWrite, ReleasedLock, SnapshotReader, TimeStamp, TxnCommitRecord},
    txn::{
        Result,
        actions::check_txn_status::{collapse_prev_rollback, make_rollback},
        commands::{
            Command, CommandExt, ReaderWithStats, ReleasedLocks, ResponsePolicy, TypedCommand,
            WriteCommand, WriteContext, WriteResult,
        },
    },
    types::SecondaryLocksStatus,
};

command! {
    /// Check secondary locks of an async commit transaction.
    ///
    /// If all prewritten locks exist, the lock information is returned.
    /// Otherwise, it returns the commit timestamp of the transaction.
    ///
    /// If the lock does not exist or is a pessimistic lock, to prevent the
    /// status being changed, a rollback may be written.
    CheckSecondaryLocks:
        cmd_ty => SecondaryLocksStatus,
        display => { "kv::command::CheckSecondaryLocks {:?} keys@{} | {:?}", (keys, start_ts, ctx), }
        content => {
            /// The keys of secondary locks.
            keys: Vec<Key>,
            /// The start timestamp of the transaction.
            start_ts: txn_types::TimeStamp,
        }
        in_heap => {
            keys,
        }
}

impl CommandExt for CheckSecondaryLocks {
    ctx!();
    tag!(check_secondary_locks);
    request_type!(KvCheckSecondaryLocks);
    ts!(start_ts);
    write_bytes!(keys: multiple);
    gen_lock!(keys: multiple);
}

#[derive(Debug, PartialEq)]
enum SecondaryLockStatus {
    Locked(Lock),
    Committed(TimeStamp),
    RolledBack,
}

// The returned `bool` indicates whether the rollback record should be written,
// it should be true if and only if the txn commit record is not found, thus
// a rollback record would be written later.
fn check_determined_txn_status<S: Snapshot>(
    reader: &mut ReaderWithStats<'_, S>,
    key: &Key,
) -> Result<(SecondaryLockStatus, bool, Option<OverlappedWrite>)> {
    match reader.get_txn_commit_record(key)? {
        TxnCommitRecord::SingleRecord { commit_ts, write } => {
            let status = if write.write_type != WriteType::Rollback {
                SecondaryLockStatus::Committed(commit_ts)
            } else {
                SecondaryLockStatus::RolledBack
            };
            // We needn't write a rollback once there is a write record for it:
            // If it's a committed record, it cannot be changed.
            // If it's a rollback record, it either comes from another
            // check_secondary_lock (thus protected) or the client stops commit
            // actively. So we don't need to make it protected again.
            Ok((status, false, None))
        }
        TxnCommitRecord::OverlappedRollback { .. } => {
            Ok((SecondaryLockStatus::RolledBack, false, None))
        }
        TxnCommitRecord::None { overlapped_write } => {
            Ok((SecondaryLockStatus::RolledBack, true, overlapped_write))
        }
    }
}

fn check_status_from_lock<S: Snapshot>(
    txn: &mut MvccTxn,
    reader: &mut ReaderWithStats<'_, S>,
    lock: Lock,
    key: &Key,
    region_id: u64,
) -> Result<(
    SecondaryLockStatus,
    bool,
    Option<OverlappedWrite>,
    Option<ReleasedLock>,
)> {
    let mut overlapped_write = None;
    if lock.is_pessimistic_lock_with_conflict() {
        assert!(lock.is_pessimistic_lock());
        let (status, need_rollback, rollback_overlapped_write) =
            check_determined_txn_status(reader, key)?;
        // If there exists commit or rollback record, the pessimistic lock is stale, in
        // this case the returned need_rollback is false.
        if !need_rollback {
            let released_lock = txn.unlock_key(key.clone(), true, TimeStamp::zero());
            return Ok((
                status,
                need_rollback,
                rollback_overlapped_write,
                released_lock,
            ));
        }
        overlapped_write = rollback_overlapped_write;
    }

    if lock.is_pessimistic_lock() {
        let released_lock = txn.unlock_key(key.clone(), true, TimeStamp::zero());
        // If the `is_pessimistic_lock_with_conflict` is true, the `overlapped_write` is
        // already fetched in the above `check_determined_txn_status` call. So
        // we don't need to fetch it again and the `overlapped_write` could be
        // reused here.
        let overlapped_write_res = if lock.is_pessimistic_lock_with_conflict() {
            overlapped_write
        } else {
            reader.get_txn_commit_record(key)?.unwrap_none(region_id)
        };
        Ok((
            SecondaryLockStatus::RolledBack,
            true,
            overlapped_write_res,
            released_lock,
        ))
    } else {
        Ok((SecondaryLockStatus::Locked(lock), false, None, None))
    }
}

impl<S: Snapshot, L: LockManager> WriteCommand<S, L> for CheckSecondaryLocks {
    fn process_write(self, snapshot: S, context: WriteContext<'_, L>) -> Result<WriteResult> {
        // It is not allowed for commit to overwrite a protected rollback. So we update
        // max_ts to prevent this case from happening.
        let region_id = self.ctx.get_region_id();
        context.concurrency_manager.update_max_ts(
            self.start_ts,
            MaxTsUpdateSource::new(|| format!("check_secondary_locks-{}", self.start_ts))
                .require_request_origin_check(self.ctx.get_request_origin()),
        )?;

        let mut txn = MvccTxn::new(self.start_ts, context.concurrency_manager);
        let mut reader = ReaderWithStats::new(
            SnapshotReader::new_with_ctx(self.start_ts, snapshot, &self.ctx),
            context.statistics,
        );
        let mut released_locks = ReleasedLocks::new();
        let mut result = SecondaryLocksStatus::Locked(Vec::new());
        let mut result_size: u64 = 0;
        for key in self.keys {
            let mut released_lock = None;
            let mut mismatch_lock = None;
            // Checks whether the given secondary lock exists.
            let (status, need_rollback, rollback_overlapped_write) = match reader.load_lock(&key)? {
                // The lock exists, the lock information is returned.
                Some(Either::Left(lock)) if lock.ts == self.start_ts => {
                    let (status, need_rollback, rollback_overlapped_write, lock_released) =
                        check_status_from_lock(&mut txn, &mut reader, lock, &key, region_id)?;
                    released_lock = lock_released;
                    (status, need_rollback, rollback_overlapped_write)
                }
                // Async commit transactions don't write shared locks, so if we get SharedLocks,
                // check the write CF for the commit record directly.
                Some(Either::Right(_)) => check_determined_txn_status(&mut reader, &key)?,
                // Searches the write CF for the commit record of the lock and returns the commit
                // timestamp (0 if the lock is not committed).
                l => {
                    // SharedLocks is already handled by the previous match arm, so this is
                    // unreachable.
                    mismatch_lock = l.map(|lock_or_shared_locks| {
                        lock_or_shared_locks
                            .left()
                            .expect("SharedLocks is handled above, should not reach here")
                    });
                    check_determined_txn_status(&mut reader, &key)?
                }
            };
            // If the lock does not exist or is a pessimistic lock, to prevent the
            // status being changed, a rollback may be written and this rollback
            // needs to be protected.
            if need_rollback {
                if let Some(l) = mismatch_lock {
                    txn.mark_rollback_on_mismatching_lock(&key, l, true);
                }
                // We must protect this rollback in case this rollback is collapsed and a stale
                // acquire_pessimistic_lock and prewrite succeed again.
                if let Some(write) = make_rollback(self.start_ts, true, rollback_overlapped_write) {
                    txn.put_write(key.clone(), self.start_ts, write.as_ref().to_bytes());
                    collapse_prev_rollback(&mut txn, &mut reader, &key)?;
                }
            }
            released_locks.push(released_lock);
            match status {
                SecondaryLockStatus::Locked(lock) => {
                    let lock_info = lock.into_lock_info(key.to_raw()?);
                    result_size += lock_info.compute_size() as u64;
                    result.push(lock_info);
                }
                SecondaryLockStatus::Committed(commit_ts) => {
                    result = SecondaryLocksStatus::Committed(commit_ts);
                    break;
                }
                SecondaryLockStatus::RolledBack => {
                    result = SecondaryLocksStatus::RolledBack;
                    break;
                }
            }
        }

        record_network_out_bytes(result_size);
        let write_result_known_txn_status =
            if let SecondaryLocksStatus::Committed(commit_ts) = &result {
                vec![(self.start_ts, *commit_ts)]
            } else {
                vec![]
            };
        let mut rows = 0;
        if let SecondaryLocksStatus::RolledBack = &result {
            // One row is mutated only when a secondary lock is rolled back.
            rows = 1;
        }
        let pr = ProcessResult::SecondaryLocksStatus { status: result };
        let new_acquired_locks = txn.take_new_locks();
        let mut write_data = WriteData::from_modifies(txn.into_modifies());
        write_data.set_allowed_on_disk_almost_full();
        Ok(WriteResult {
            ctx: self.ctx,
            to_be_write: write_data,
            rows,
            pr,
            lock_info: vec![],
            released_locks,
            new_acquired_locks,
            lock_guards: vec![],
            response_policy: ResponsePolicy::OnApplied,
            known_txn_status: write_result_known_txn_status,
        })
    }
}

#[cfg(test)]
pub mod tests {
    use std::sync::Arc;

    use concurrency_manager::ConcurrencyManager;
    use kvproto::kvrpcpb::Context;
    use tikv_util::deadline::Deadline;

    use super::*;
    use crate::storage::{
        Engine,
        kv::TestEngineBuilder,
        lock_manager::MockLockManager,
        mvcc::tests::*,
        txn::{
            commands::WriteCommand, scheduler::DEFAULT_EXECUTION_DURATION_LIMIT, tests::*,
            txn_status_cache::TxnStatusCache,
        },
    };

    pub fn must_success<E: Engine>(
        engine: &mut E,
        key: &[u8],
        lock_ts: impl Into<TimeStamp>,
        expect_status: SecondaryLocksStatus,
    ) {
        let ctx = Context::default();
        let snapshot = engine.snapshot(Default::default()).unwrap();
        let lock_ts = lock_ts.into();
        let cm = ConcurrencyManager::new(lock_ts);
        let command = crate::storage::txn::commands::CheckSecondaryLocks {
            ctx: ctx.clone(),
            keys: vec![Key::from_raw(key)],
            start_ts: lock_ts,
            deadline: Deadline::from_now(DEFAULT_EXECUTION_DURATION_LIMIT),
        };
        let result = command
            .process_write(
                snapshot,
                WriteContext {
                    lock_mgr: &MockLockManager::new(),
                    concurrency_manager: cm,
                    extra_op: Default::default(),
                    statistics: &mut Default::default(),
                    async_apply_prewrite: false,
                    raw_ext: None,
                    txn_status_cache: Arc::new(TxnStatusCache::new_for_test()),
                },
            )
            .unwrap();
        if let ProcessResult::SecondaryLocksStatus { status } = result.pr {
            assert_eq!(status, expect_status);
            write(engine, &ctx, result.to_be_write.modifies);
        } else {
            unreachable!();
        }
    }

    #[test]
    fn test_check_async_commit_secondary_locks() {
        let mut engine = TestEngineBuilder::new().build().unwrap();
        let mut engine_clone = engine.clone();
        let ctx = Context::default();
        let cm = ConcurrencyManager::new(1.into());

        let mut check_secondary = |key, ts| {
            let snapshot = engine_clone.snapshot(Default::default()).unwrap();
            let key = Key::from_raw(key);
            let ts = TimeStamp::new(ts);
            let command = crate::storage::txn::commands::CheckSecondaryLocks {
                ctx: Default::default(),
                keys: vec![key],
                start_ts: ts,
                deadline: Deadline::from_now(DEFAULT_EXECUTION_DURATION_LIMIT),
            };
            let result = command
                .process_write(
                    snapshot,
                    WriteContext {
                        lock_mgr: &MockLockManager::new(),
                        concurrency_manager: cm.clone(),
                        extra_op: Default::default(),
                        statistics: &mut Default::default(),
                        async_apply_prewrite: false,
                        raw_ext: None,
                        txn_status_cache: Arc::new(TxnStatusCache::new_for_test()),
                    },
                )
                .unwrap();
            if !result.to_be_write.modifies.is_empty() {
                engine_clone.write(&ctx, result.to_be_write).unwrap();
            }
            if let ProcessResult::SecondaryLocksStatus { status } = result.pr {
                status
            } else {
                unreachable!();
            }
        };

        must_prewrite_lock(&mut engine, b"k1", b"key", 1);
        must_commit(&mut engine, b"k1", 1, 3);
        must_rollback(&mut engine, b"k1", 5, false);
        must_prewrite_lock(&mut engine, b"k1", b"key", 7);
        must_commit(&mut engine, b"k1", 7, 9);

        // Lock CF has no lock
        //
        // LOCK CF       | WRITE CF
        // --------------+---------------------
        //               | 9: start_ts = 7
        //               | 5: rollback
        //               | 3: start_ts = 1

        assert_eq!(
            check_secondary(b"k1", 7),
            SecondaryLocksStatus::Committed(9.into())
        );
        must_get_commit_ts(&mut engine, b"k1", 7, 9);
        assert_eq!(check_secondary(b"k1", 5), SecondaryLocksStatus::RolledBack);
        must_get_rollback_ts(&mut engine, b"k1", 5);
        assert_eq!(
            check_secondary(b"k1", 1),
            SecondaryLocksStatus::Committed(3.into())
        );
        must_get_commit_ts(&mut engine, b"k1", 1, 3);
        assert_eq!(check_secondary(b"k1", 6), SecondaryLocksStatus::RolledBack);
        must_get_rollback_protected(&mut engine, b"k1", 6, true);

        // ----------------------------

        must_acquire_pessimistic_lock(&mut engine, b"k1", b"key", 11, 11);

        // Lock CF has a pessimistic lock
        //
        // LOCK CF       | WRITE CF
        // ------------------------------------
        // ts = 11 (pes) | 9: start_ts = 7
        //               | 5: rollback
        //               | 3: start_ts = 1

        let status = check_secondary(b"k1", 11);
        assert_eq!(status, SecondaryLocksStatus::RolledBack);
        must_get_rollback_protected(&mut engine, b"k1", 11, true);

        // ----------------------------

        must_prewrite_lock(&mut engine, b"k1", b"key", 13);

        // Lock CF has an optimistic lock
        //
        // LOCK CF       | WRITE CF
        // ------------------------------------
        // ts = 13 (opt) | 11: rollback
        //               |  9: start_ts = 7
        //               |  5: rollback
        //               |  3: start_ts = 1

        match check_secondary(b"k1", 13) {
            SecondaryLocksStatus::Locked(_) => {}
            res => panic!("unexpected lock status: {:?}", res),
        }
        must_locked(&mut engine, b"k1", 13);

        // ----------------------------

        must_commit(&mut engine, b"k1", 13, 15);

        // Lock CF has an optimistic lock
        //
        // LOCK CF       | WRITE CF
        // ------------------------------------
        //               | 15: start_ts = 13
        //               | 11: rollback
        //               |  9: start_ts = 7
        //               |  5: rollback
        //               |  3: start_ts = 1

        match check_secondary(b"k1", 14) {
            SecondaryLocksStatus::RolledBack => {}
            res => panic!("unexpected lock status: {:?}", res),
        }
        must_get_rollback_protected(&mut engine, b"k1", 14, true);

        match check_secondary(b"k1", 15) {
            SecondaryLocksStatus::RolledBack => {}
            res => panic!("unexpected lock status: {:?}", res),
        }
        must_get_overlapped_rollback(&mut engine, b"k1", 15, 13, WriteType::Lock, Some(0));

        // Lock CF has an stale pessimistic lock, the transaction is already committed
        // or rolled back.
        //
        // LOCK CF       | WRITE CF
        // ------------------------------------
        //               | 15: start_ts = 13 with overlapped rollback
        //               | 14: rollback
        //               | 11: rollback
        //               |  9: start_ts = 7
        //               |  5: rollback
        //               |  3: start_ts = 1
        must_acquire_pessimistic_lock_allow_lock_with_conflict(
            &mut engine,
            b"k1",
            b"key",
            7,
            7,
            true,
            false,
            10,
        )
        .assert_locked_with_conflict(None, 15);
        match check_secondary(b"k1", 7) {
            SecondaryLocksStatus::Committed(ts) => {
                assert!(ts.eq(&9.into()));
            }
            res => panic!("unexpected lock status: {:?}", res),
        }
        must_unlocked(&mut engine, b"k1");

        // Lock CF has an pessimistic lock, the transaction status is not found
        // in storage.
        must_acquire_pessimistic_lock_allow_lock_with_conflict(
            &mut engine,
            b"k1",
            b"key",
            8,
            8,
            true,
            false,
            10,
        )
        .assert_locked_with_conflict(None, 15);
        match check_secondary(b"k1", 8) {
            SecondaryLocksStatus::RolledBack => {}
            res => panic!("unexpected lock status: {:?}", res),
        }
        must_unlocked(&mut engine, b"k1");
    }

    #[test]
    #[ignore = "requires the rust-rocksdb ingest-write-bug hook branch"]
    fn test_ingest_allow_write_can_make_async_commit_recovery_rollback_prewrite() {
        use std::{env, fs, thread, time::Duration};

        use engine_rocks::{RocksSstWriterBuilder, raw::IngestExternalFileOptions, util};
        use engine_traits::{
            CF_DEFAULT, CF_LOCK, CF_WRITE, KvEngine, MiscExt, Peekable, SnapshotMiscExt, SstWriter,
            SstWriterBuilder,
        };
        use kvproto::kvrpcpb::AssertionLevel;
        use tempfile::Builder;
        use txn_types::Mutation;

        use crate::storage::{
            kv::Modify,
            mvcc::SHORT_VALUE_MAX_LEN,
            txn::commands::{Prewrite, TypedCommand},
        };

        const INGEST_SEQUENCE_PAUSE_ENV: &str = "TIKV_REPRO_19891_INGEST_AFTER_LAST_SEQUENCE_MS";
        const INGEST_SEQUENCE_MARKER_ENV: &str =
            "TIKV_REPRO_19891_INGEST_AFTER_LAST_SEQUENCE_MARKER";

        env::set_var(INGEST_SEQUENCE_PAUSE_ENV, "500");

        let path_dir = Builder::new()
            .prefix("test_async_commit_recovery_false_rollback")
            .tempdir()
            .unwrap();
        let marker_path = path_dir.path().join("after-last-sequence.marker");
        env::set_var(INGEST_SEQUENCE_MARKER_ENV, marker_path.to_str().unwrap());

        let mut engine = TestEngineBuilder::new()
            .path(path_dir.path().join("db"))
            .build()
            .unwrap();
        let rocks_db = engine.get_rocksdb();

        let dummy_key = b"k0";
        let target_key = b"k1";
        let primary_key = b"primary-on-other-region";
        let start_ts = TimeStamp::new(10);
        let read_ts = TimeStamp::new(50);
        let long_value = vec![b'v'; SHORT_VALUE_MAX_LEN + 1];
        let mutations = vec![
            Mutation::make_put(Key::from_raw(dummy_key), long_value.clone()),
            Mutation::make_put(Key::from_raw(target_key), long_value.clone()),
        ];
        let prewrite_cmd = Prewrite::new(
            mutations,
            primary_key.to_vec(),
            start_ts,
            0,
            false,
            2,
            TimeStamp::zero(),
            TimeStamp::new(100),
            Some(vec![]),
            false,
            AssertionLevel::Off,
            Context::default(),
        );

        let prewrite_snapshot = engine.snapshot(Default::default()).unwrap();
        let prewrite_lock_mgr = MockLockManager::new();
        let mut prewrite_statistics = Default::default();
        let prewrite_result = prewrite_cmd
            .cmd
            .process_write(
                prewrite_snapshot,
                WriteContext {
                    lock_mgr: &prewrite_lock_mgr,
                    concurrency_manager: ConcurrencyManager::new(start_ts),
                    extra_op: Default::default(),
                    statistics: &mut prewrite_statistics,
                    async_apply_prewrite: false,
                    raw_ext: None,
                    txn_status_cache: Arc::new(TxnStatusCache::new_for_test()),
                },
            )
            .unwrap();
        match &prewrite_result.pr {
            ProcessResult::PrewriteResult { result } => {
                assert!(result.locks.is_empty());
                assert!(!result.min_commit_ts.is_zero());
                assert_eq!(result.one_pc_commit_ts, TimeStamp::zero());
            }
            other => panic!("unexpected prewrite result: {:?}", other),
        }

        let prewrite_modifies = prewrite_result.to_be_write.modifies;
        let expected_dummy_default = Key::from_raw(dummy_key).append_ts(start_ts);
        let expected_dummy_lock = Key::from_raw(dummy_key);
        let expected_target_default = Key::from_raw(target_key).append_ts(start_ts);
        let expected_target_lock = Key::from_raw(target_key);
        match prewrite_modifies.as_slice() {
            [
                Modify::Put(cf0, key0, _),
                Modify::Put(cf1, key1, _),
                Modify::Put(cf2, key2, _),
                Modify::Put(cf3, key3, _),
            ] => {
                assert_eq!(*cf0, CF_DEFAULT);
                assert_eq!(key0, &expected_dummy_default);
                assert_eq!(*cf1, CF_LOCK);
                assert_eq!(key1, &expected_dummy_lock);
                assert_eq!(*cf2, CF_DEFAULT);
                assert_eq!(key2, &expected_target_default);
                assert_eq!(*cf3, CF_LOCK);
                assert_eq!(key3, &expected_target_lock);
            }
            other => panic!("unexpected async prewrite modifies: {:?}", other),
        }

        let sst_path = path_dir.path().join("ingest.sst");
        let mut sst = RocksSstWriterBuilder::new()
            .set_db(&rocks_db)
            .set_cf(CF_DEFAULT)
            .build(sst_path.to_str().unwrap())
            .unwrap();
        sst.put(b"ingest-key", b"ingest-value").unwrap();
        sst.finish().unwrap();

        // The ingest-write-bug rust-rocksdb branch pauses after RocksDB reads
        // VersionSet::LastSequence and before it publishes the sequence consumed
        // by the ingested file.
        let _pre_ingest_snapshot = rocks_db.snapshot();
        let ingest_db = rocks_db.clone();
        let ingest_path = sst_path.to_str().unwrap().to_owned();
        let ingest_thread = thread::spawn(move || {
            let cf = util::get_cf_handle(ingest_db.as_inner(), CF_DEFAULT).unwrap();
            let mut opts = IngestExternalFileOptions::new();
            opts.move_files(true);
            opts.snapshot_consistent(true);
            opts.allow_global_seqno(true);
            opts.set_write_global_seqno(false);
            opts.set_allow_write(true);
            opts.allow_blocking_flush(true);
            ingest_db
                .as_inner()
                .ingest_external_file_cf(cf, &opts, &[ingest_path.as_str()])
                .unwrap()
        });

        let wait_started = std::time::Instant::now();
        while !marker_path.exists() {
            assert!(
                wait_started.elapsed() < Duration::from_secs(10),
                "RocksDB did not enter the ingest sequence update hook"
            );
            thread::sleep(Duration::from_millis(10));
        }

        write(&engine, &Context::default(), prewrite_modifies);
        let seq_after_prewrite = rocks_db.get_latest_sequence_number();

        ingest_thread.join().unwrap();
        env::remove_var(INGEST_SEQUENCE_PAUSE_ENV);
        env::remove_var(INGEST_SEQUENCE_MARKER_ENV);
        let _ = fs::remove_file(&marker_path);

        let seq_after_ingest = rocks_db.get_latest_sequence_number();
        assert!(
            seq_after_ingest < seq_after_prewrite,
            "ingest should regress latest sequence below the acknowledged async prewrite batch: after_ingest={}, after_prewrite={}",
            seq_after_ingest,
            seq_after_prewrite
        );

        let regressed_snapshot = rocks_db.snapshot();
        assert_eq!(regressed_snapshot.sequence_number(), seq_after_ingest);
        assert!(
            regressed_snapshot
                .get_value_cf(CF_DEFAULT, expected_dummy_default.as_encoded())
                .unwrap()
                .is_some(),
            "the first record of the acknowledged async prewrite batch is visible"
        );
        assert!(
            regressed_snapshot
                .get_value_cf(CF_LOCK, expected_target_lock.as_encoded())
                .unwrap()
                .is_none(),
            "the target secondary lock is hidden by the regressed sequence"
        );
        assert!(
            regressed_snapshot
                .get_value_cf(
                    CF_WRITE,
                    expected_target_lock.append_ts(start_ts).as_encoded()
                )
                .unwrap()
                .is_none(),
            "the target secondary has no commit or rollback record yet"
        );
        must_get_none(&mut engine, target_key, read_ts);

        let check_snapshot = engine.snapshot(Default::default()).unwrap();
        let check_lock_mgr = MockLockManager::new();
        let mut check_statistics = Default::default();
        let check_cmd: TypedCommand<SecondaryLocksStatus> = CheckSecondaryLocks::new(
            vec![Key::from_raw(target_key)],
            start_ts,
            Context::default(),
        );
        let check_result = check_cmd
            .cmd
            .process_write(
                check_snapshot,
                WriteContext {
                    lock_mgr: &check_lock_mgr,
                    concurrency_manager: ConcurrencyManager::new(start_ts),
                    extra_op: Default::default(),
                    statistics: &mut check_statistics,
                    async_apply_prewrite: false,
                    raw_ext: None,
                    txn_status_cache: Arc::new(TxnStatusCache::new_for_test()),
                },
            )
            .unwrap();
        match &check_result.pr {
            ProcessResult::SecondaryLocksStatus { status } => {
                assert_eq!(*status, SecondaryLocksStatus::RolledBack);
            }
            other => panic!("unexpected check-secondary result: {:?}", other),
        }
        assert!(
            check_result
                .to_be_write
                .modifies
                .iter()
                .any(|m| matches!(m, Modify::Put(CF_WRITE, key, _) if key == &Key::from_raw(target_key).append_ts(start_ts))),
            "CheckSecondaryLocks should persist a rollback record for the hidden secondary lock: {:?}",
            check_result.to_be_write.modifies
        );

        write(
            &engine,
            &Context::default(),
            check_result.to_be_write.modifies,
        );
        must_get_rollback_protected(&mut engine, target_key, start_ts, true);

        // The recovery command has made a durable rollback decision for the
        // acknowledged async prewrite. Even if later sequence advancement makes
        // the hidden lock visible again, there is still no committed version for
        // this mutation.
        let read_snapshot = engine.snapshot(Default::default()).unwrap();
        let mut reader = SnapshotReader::new(read_ts, read_snapshot, true);
        assert_eq!(
            reader.get(&Key::from_raw(target_key), read_ts).unwrap(),
            None,
            "the acknowledged async prewrite value should not be readable after the false rollback"
        );
    }
}
