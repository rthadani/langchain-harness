#!/bin/sh
# Pass-through entrypoint. Maps the de-facto PORT variable that many platforms
# inject onto HARNESS_SERVER_PORT, then defers to the harness CLI.
set -eu

if [ -n "${PORT:-}" ] && [ -z "${HARNESS_SERVER_PORT:-}" ]; then
    export HARNESS_SERVER_PORT="$PORT"
fi

if [ "$#" -eq 0 ]; then
    set -- serve
fi

exec harness "$@"
