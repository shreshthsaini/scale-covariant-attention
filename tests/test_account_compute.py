"""Resource accounting must not double count copied logs, retries, or shared nodes."""
from scripts.account_compute import pair_events, parse_events, union_seconds


def test_duplicate_logs_and_retry_attempts():
    text = '\n'.join([
        '2026-10-16T23:59:30 c609-092-guest start a.sh',
        '2026-10-17T00:00:30 c609-092-guest end a.sh rc=1',
        '2026-10-17T00:00:30 c609-092-guest start a.sh',
        '2026-10-17T00:02:30 c609-092-guest end a.sh rc=0',
        '2026-10-17T00:02:30 c609-092-guest start unfinished.sh',
    ])
    a = parse_events(text, 'guest_original.log')
    b = parse_events(text, 'guest_copy.log')
    spans, censored, orphan, duplicates = pair_events(a + b)
    assert len(spans) == 2 and [s['rc'] for s in spans] == ['1', '0']
    assert len(censored) == 1 and not orphan and duplicates == 5
    assert union_seconds(spans) == 180
    assert union_seconds(spans, conservative=True) == 178


def test_shared_node_spans_are_unioned():
    spans = [dict(host='node1', start_epoch=0, end_epoch=100),
             dict(host='node1', start_epoch=50, end_epoch=150),
             dict(host='node2', start_epoch=50, end_epoch=150)]
    assert union_seconds(spans) == 250


def test_zero_second_task_and_orphan_end():
    text = '\n'.join([
        '2026-10-17T00:00:00 c609-092-guest start instant.sh',
        '2026-10-17T00:00:00 c609-092-guest end instant.sh rc=0',
        '2026-10-17T00:01:00 c609-092-guest end unknown.sh rc=0',
    ])
    spans, censored, orphan, _ = pair_events(parse_events(text, 'guest.log'))
    assert len(spans) == 1 and spans[0]['seconds'] == 0
    assert not censored and len(orphan) == 1


def test_preemptible_worker_logs():
    text = '\n'.join([
        '2026-10-17T00:00:00 start j000_area_krea.sh.c609-092-guest-safe',
        '2026-10-17T00:01:00 stale j000_area_krea.sh',
    ])
    spans, censored, orphan, _ = pair_events(parse_events(text, 'preemptible_c609-092.log'))
    assert len(spans) == 1 and spans[0]['seconds'] == 60
    assert spans[0]['outcome'] == 'stale'
    assert not censored and not orphan
