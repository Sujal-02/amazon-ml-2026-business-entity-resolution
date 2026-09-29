#!/usr/bin/env bash
# Logs RAM / disk / load every 60 s to $WORK/logs/resources.log and syncs logs to S3 every 5 min.
WORK=${WORK_DIR:-/data/work}; S3=${S3_PREFIX:-s3://ber-ml-170848441936-aps1/worker}
export PATH=/snap/bin:$PATH
mkdir -p "$WORK/logs"
i=0
while true; do
  echo "$(date +%F_%T) load=$(cut -d' ' -f1-3 /proc/loadavg) mem_used_mb=$(free -m | awk '/Mem:/{print $3}') mem_avail_mb=$(free -m | awk '/Mem:/{print $7}') disk_used=$(df -h /data | awk 'NR==2{print $5}') top=$(ps -eo comm,%cpu,rss --sort=-rss | sed -n 2p | tr -s ' ')" >> "$WORK/logs/resources.log"
  i=$((i + 1)); [ $((i % 5)) -eq 0 ] && aws s3 sync "$WORK/logs" "$S3/logs" --only-show-errors
  sleep 60
done
