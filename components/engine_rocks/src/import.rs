// Copyright 2019 TiKV Project Authors. Licensed under Apache-2.0.

use engine_traits::{ImportExt, IngestExternalFileOptions, Range, Result};
use fail::fail_point;
use rocksdb::IngestExternalFileOptions as RawIngestExternalFileOptions;
use tikv_util::{range_latch::RangeLatchGuard, time::Instant};

use crate::{
    engine::RocksEngine,
    perf_context_metrics::{
        INGEST_EXTERNAL_FILE_ALLOW_WRITE_COUNTER, INGEST_EXTERNAL_FILE_TIME_HISTOGRAM,
    },
    r2e, util,
};

// Temporarily disabled due to https://github.com/tikv/tikv/issues/19891.
const ENABLE_INGEST_ALLOW_WRITE: bool = false;

fn should_allow_write_during_ingest(range: Option<&Range<'_>>, force_allow_write: bool) -> bool {
    ENABLE_INGEST_ALLOW_WRITE && (range.is_some() || force_allow_write)
}

impl ImportExt for RocksEngine {
    type IngestExternalFileOptions = RocksIngestExternalFileOptions;

    fn ingest_external_file_cf(
        &self,
        cf_name: &str,
        files: &[&str],
        range: Option<Range<'_>>,
        force_allow_write: bool,
    ) -> Result<()> {
        // Acquire latch to prevent concurrency with compaction-filter operations
        // when using RocksDB IngestExternalFileOptions.allow_write = true.
        let _region_inject_latch_guard = range.as_ref().map(|r| {
            self.ingest_latch
                .acquire(r.start_key.to_vec(), r.end_key.to_vec())
        });
        fail_point!("after_apply_snapshot_ingest_latch_acquired");

        let cf = util::get_cf_handle(self.as_inner(), cf_name)?;
        let mut opts = RocksIngestExternalFileOptions::new();
        opts.move_files(true);
        opts.set_write_global_seqno(false);
        let allow_write = should_allow_write_during_ingest(range.as_ref(), force_allow_write);
        opts.allow_write(allow_write);
        if allow_write {
            INGEST_EXTERNAL_FILE_ALLOW_WRITE_COUNTER
                .with_label_values(&["
            allow_write"])
                .inc();
        } else {
            INGEST_EXTERNAL_FILE_ALLOW_WRITE_COUNTER
                .with_label_values(&["
            not_allow_write"])
                .inc();
        }

        // Note: no need reset the global seqno to 0 for compatibility as #16992
        // enable the TiKV to handle the case on applying abnormal snapshot.
        let now = Instant::now_coarse();
        // This is calling a specially optimized version of
        // ingest_external_file_cf. In cases where the memtable needs to be
        // flushed it avoids blocking writers while doing the flush. The
        // return value here just indicates whether the fallback path requiring
        // the manual memtable flush was taken.
        let did_memtable_flush = self
            .as_inner()
            .ingest_external_file_optimized(cf, &opts.0, files)
            .map_err(r2e)?;
        let time_cost = now.saturating_elapsed_secs();
        if did_memtable_flush {
            INGEST_EXTERNAL_FILE_TIME_HISTOGRAM
                .get(cf_name.into())
                .block
                .observe(time_cost);
        } else {
            INGEST_EXTERNAL_FILE_TIME_HISTOGRAM
                .get(cf_name.into())
                .non_block
                .observe(time_cost);
        }
        Ok(())
    }

    fn acquire_ingest_latch(&self, range: Range<'_>) -> RangeLatchGuard<'_> {
        self.ingest_latch
            .acquire(range.start_key.to_vec(), range.end_key.to_vec())
    }
}

pub struct RocksIngestExternalFileOptions(RawIngestExternalFileOptions);

impl IngestExternalFileOptions for RocksIngestExternalFileOptions {
    fn new() -> RocksIngestExternalFileOptions {
        RocksIngestExternalFileOptions(RawIngestExternalFileOptions::new())
    }

    fn move_files(&mut self, f: bool) {
        self.0.move_files(f);
    }

    fn allow_write(&mut self, f: bool) {
        self.0.set_allow_write(f);
    }

    fn get_write_global_seqno(&self) -> bool {
        self.0.get_write_global_seqno()
    }

    fn set_write_global_seqno(&mut self, f: bool) {
        self.0.set_write_global_seqno(f);
    }
}

#[cfg(test)]
mod tests {
    use std::{env, fs, thread, time::Duration};

    use engine_traits::{
        ALL_CFS, CF_DEFAULT, CF_LOCK, CF_WRITE, FlowControlFactorsExt, KvEngine, MiscExt, Mutable,
        Peekable, SnapshotMiscExt, SstWriter, SstWriterBuilder, SyncMutable, WriteBatch,
        WriteBatchExt,
    };
    use tempfile::Builder;
    use txn_types::{Key, Lock, LockType, TimeStamp, Write, WriteType};

    use super::*;
    use crate::{RocksCfOptions, RocksDbOptions, RocksSstWriterBuilder, util::new_engine_opt};

    const INGEST_SEQUENCE_PAUSE_ENV: &str = "TIKV_REPRO_19891_INGEST_AFTER_LAST_SEQUENCE_MS";
    const INGEST_SEQUENCE_MARKER_ENV: &str = "TIKV_REPRO_19891_INGEST_AFTER_LAST_SEQUENCE_MARKER";

    #[test]
    fn test_ingest_allow_write_is_disabled() {
        let range = Range::new(b"key", b"kez");

        // #19891 is mitigated by keeping RocksDB allow_write disabled even for
        // snapshot-like ranged ingests and explicit force_allow_write requests.
        assert!(!should_allow_write_during_ingest(None, false));
        assert!(!should_allow_write_during_ingest(None, true));
        assert!(!should_allow_write_during_ingest(Some(&range), false));
        assert!(!should_allow_write_during_ingest(Some(&range), true));
    }

    #[test]
    #[ignore = "requires a RocksDB hook that pauses after IngestExternalFiles reads LastSequence"]
    fn test_ingest_allow_write_can_expose_committed_write_with_stale_lock() {
        env::set_var(INGEST_SEQUENCE_PAUSE_ENV, "500");

        let path_dir = Builder::new()
            .prefix("test_ingest_allow_write_sequence_regression")
            .tempdir()
            .unwrap();
        let root_path = path_dir.path();
        let db_path = root_path.join("db");
        let path_str = db_path.to_str().unwrap();
        let marker_path = root_path.join("after-last-sequence.marker");
        env::set_var(INGEST_SEQUENCE_MARKER_ENV, marker_path.to_str().unwrap());

        let cfs_opts = ALL_CFS
            .iter()
            .map(|cf| (*cf, RocksCfOptions::default()))
            .collect();
        let db = new_engine_opt(path_str, RocksDbOptions::default(), cfs_opts).unwrap();

        let raw_key = b"k1";
        let encoded_key = Key::from_raw(raw_key);
        let start_ts = TimeStamp::compose(5, 0);
        let commit_ts = TimeStamp::compose(10, 0);
        let min_commit_ts = TimeStamp::compose(15, 0);
        let lock = Lock::new(
            LockType::Put,
            raw_key.to_vec(),
            start_ts,
            10,
            Some(b"v1".to_vec()),
            start_ts,
            1,
            min_commit_ts,
            false,
        );
        db.put_cf(CF_LOCK, encoded_key.as_encoded(), &lock.to_bytes())
            .unwrap();

        let sst_path = root_path.join("ingest.sst");
        let mut sst = RocksSstWriterBuilder::new()
            .set_db(&db)
            .set_cf(CF_DEFAULT)
            .build(sst_path.to_str().unwrap())
            .unwrap();
        sst.put(b"ingest-key", b"ingest-value").unwrap();
        sst.finish().unwrap();

        // Holding a snapshot forces RocksDB to assign a global sequence number
        // to the ingested file. That makes DBImpl update VersionSet's last
        // sequence after LogAndApply, which is the race window for #19891.
        let _pre_ingest_snapshot = db.snapshot();
        let ingest_db = db.clone();
        let ingest_path = sst_path.to_str().unwrap().to_owned();
        let ingest_thread = thread::spawn(move || {
            let cf = util::get_cf_handle(ingest_db.as_inner(), CF_DEFAULT).unwrap();
            let mut opts = RocksIngestExternalFileOptions::new();
            opts.move_files(true);
            opts.0.snapshot_consistent(true);
            opts.0.allow_global_seqno(true);
            opts.set_write_global_seqno(false);
            opts.allow_write(true);
            ingest_db
                .as_inner()
                .ingest_external_file_optimized(cf, &opts.0, &[ingest_path.as_str()])
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

        // The operation order in this WriteBatch matters. A regressed snapshot
        // at the ingested file's assigned seqno can see the committed write,
        // but cannot see the following lock delete.
        let write_record = Write::new(WriteType::Put, start_ts, Some(b"v1".to_vec()))
            .as_ref()
            .to_bytes();
        let write_key = Key::from_raw(raw_key).append_ts(commit_ts);
        let mut wb = db.write_batch();
        wb.put_cf(CF_WRITE, write_key.as_encoded(), &write_record)
            .unwrap();
        wb.delete_cf(CF_LOCK, encoded_key.as_encoded()).unwrap();
        wb.write().unwrap();
        let seq_after_commit_batch = db.get_latest_sequence_number();

        ingest_thread.join().unwrap();
        env::remove_var(INGEST_SEQUENCE_PAUSE_ENV);
        env::remove_var(INGEST_SEQUENCE_MARKER_ENV);
        let _ = fs::remove_file(&marker_path);

        let seq_after_ingest = db.get_latest_sequence_number();
        assert!(
            seq_after_ingest < seq_after_commit_batch,
            "ingest should overwrite the foreground write sequence: after_ingest={}, after_commit={}",
            seq_after_ingest,
            seq_after_commit_batch
        );

        let snapshot = db.snapshot();
        assert_eq!(snapshot.sequence_number(), seq_after_ingest);
        assert!(
            snapshot
                .get_value_cf(CF_WRITE, write_key.as_encoded())
                .unwrap()
                .is_some(),
            "the commit record is visible at the regressed snapshot"
        );
        assert!(
            snapshot
                .get_value_cf(CF_LOCK, encoded_key.as_encoded())
                .unwrap()
                .is_some(),
            "the lock delete is hidden at the regressed snapshot"
        );
    }

    #[test]
    fn test_ingest_multiple_file() {
        let path_dir = Builder::new()
            .prefix("test_ingest_multiple_file")
            .tempdir()
            .unwrap();
        let root_path = path_dir.path();
        let db_path = root_path.join("db");
        let path_str = db_path.to_str().unwrap();

        let cfs_opts = ALL_CFS
            .iter()
            .map(|cf| {
                let mut opt = RocksCfOptions::default();
                opt.set_force_consistency_checks(true);
                (*cf, opt)
            })
            .collect();
        let db = new_engine_opt(path_str, RocksDbOptions::default(), cfs_opts).unwrap();
        let mut wb = db.write_batch();
        for i in 1000..5000 {
            let v = i.to_string();
            wb.put(v.as_bytes(), v.as_bytes()).unwrap();
            if i % 1000 == 100 {
                wb.write().unwrap();
                wb.clear();
            }
        }
        // Flush one memtable to L0 to make sure that the next sst files to be ingested
        //  must locate in L0.
        db.flush_cf(CF_DEFAULT, true).unwrap();
        assert_eq!(
            1,
            db.get_cf_num_files_at_level(CF_DEFAULT, 0)
                .unwrap()
                .unwrap()
        );

        let p1 = root_path.join("sst1");
        let p2 = root_path.join("sst2");
        let mut sst1 = RocksSstWriterBuilder::new()
            .set_db(&db)
            .set_cf(CF_DEFAULT)
            .build(p1.to_str().unwrap())
            .unwrap();
        let mut sst2 = RocksSstWriterBuilder::new()
            .set_db(&db)
            .set_cf(CF_DEFAULT)
            .build(p2.to_str().unwrap())
            .unwrap();
        for i in 1001..2000 {
            let v = i.to_string();
            sst1.put(v.as_bytes(), v.as_bytes()).unwrap();
        }
        sst1.finish().unwrap();
        for i in 2001..3000 {
            let v = i.to_string();
            sst2.put(v.as_bytes(), v.as_bytes()).unwrap();
        }
        sst2.finish().unwrap();
        db.ingest_external_file_cf(
            CF_DEFAULT,
            &[p1.to_str().unwrap(), p2.to_str().unwrap()],
            None,
            false, // force_allow_write
        )
        .unwrap();
    }
}
