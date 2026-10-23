---
name: network-diagnostics
description: How to run latency, packet loss, DNS, and download speed tests to diagnose slow internet, plus how to interpret the results and recommend fixes. Use when the user reports slow internet or network problems.
---

# Network Diagnostics

Run a structured set of network tests, then interpret the results and recommend fixes.

## Important caveat

These tests run in an isolated sandbox. Measurements may not reflect the user's actual machine or network. State this caveat up front, and recommend the user confirm findings with a browser-based test such as fast.com or speedtest.net on their own machine.

## Test procedure

Run these tests in order. Batch independent commands together where possible. Each command needs a timeout of at least 30 seconds.

### 1. Latency and packet loss

Ping two public DNS servers to measure round-trip time and packet loss:

```
ping -c 4 8.8.8.8
ping -c 4 1.1.1.1
```

**Healthy:** under ~50 ms average, 0% packet loss. Packet loss above 0% is a red flag for a flaky connection (Wi-Fi interference, ISP issues, failing hardware).

### 2. DNS and connection timing

Measure the phases of an HTTPS request:

```
curl -o /dev/null -s -w "DNS: %{time_namelookup}s | Connect: %{time_connect}s | Total: %{time_total}s\n" https://www.google.com
```

**Healthy:** DNS under ~0.1 s, connect under ~0.15 s. Slow DNS with fast ping suggests the user's resolver is the problem (recommend switching to 1.1.1.1 or 8.8.8.8).

### 3. Download throughput

Fetch a known-size payload:

```
curl -o /dev/null -s -w "Downloaded 10MB in %{time_total}s => %{speed_download} bytes/s\n" https://speed.cloudflare.com/__down?bytes=10000000
```

Convert bytes/s to Mbps (multiply by 8, divide by 1,000,000) and report both. Run it twice if the first result looks anomalous.

## Interpreting results

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Packet loss > 0% | Flaky connection (Wi-Fi interference, ISP issue, failing hardware) | Move closer to router; try 5 GHz or Ethernet; contact ISP if persistent |
| High latency (> 100 ms) | Distance/routing to server, or congestion | Test a closer server; check for bandwidth contention |
| Slow DNS, fast ping | Client-side resolver is slow | Switch DNS to 1.1.1.1 or 8.8.8.8 |
| Slow throughput, fast ping | Bandwidth contention or ISP throttling | Pause background syncs/downloads; check other devices; test at different times |
| Everything healthy but user reports slowness | Local issue: Wi-Fi signal, router needs restart, VPN/proxy, browser bloat | Restart router; disable VPN temporarily; test in incognito; reduce tabs/extensions |

## Report format

Present results as a short table, then interpret:

1. **Verdict:** is the measured connection healthy or not?
2. **Likely cause** if the user still experiences slowness despite healthy results.
3. **Recommended fixes** ordered from easiest to hardest.
4. **Comparison ask:** ask the user to run fast.com or speedtest.net on their machine and share the number so results can be compared against the sandbox measurement.

Be concise. Never fabricate numbers — if a command fails, report the failure and continue with the tests that work.
