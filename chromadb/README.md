## 백업
crontab -e
### 매일 10시에 백업
0 10 * * * /bin/bash -c 'TODAY=$(date +\%Y-\%m-\%d); SRC_DIR="/home/deploy/chroma-data"; BACKUP_DIR="/home/deploy/chromadb-backup"; mkdir -p 

### 최대 14개로 유지
"$BACKUP_DIR"; tar -czf "$BACKUP_DIR/chroma-data-$TODAY.tar.gz" -C "$SRC_DIR" .'
5 10 * * * cd /home/deploy/chromadb-backup && ls -1tr | head -n -14 | xargs -d '\n' rm -f --


## 복구 
# 1. 컨테이너 중지
docker compose -f /home/deploy/2-hertz-ai/docker-compose.yml stop chromadb

# 2. 기존 데이터 삭제 (필요한 경우만!)
rm -rf /home/deploy/chroma-data/*

# 3. 백업 파일 압축 해제
tar -xzf /home/deploy/chromadb-backup/chroma-data-2025-05-28.tar.gz -C /home/deploy/chroma-data

# 4. 컨테이너 시작
docker compose -f /home/deploy/2-hertz-ai/docker-compose.yml start chromadb