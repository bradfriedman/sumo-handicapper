"""
Quick data exploration script for sumo database
"""
import pymysql
import pandas as pd

# Database connection
conn = pymysql.connect(
    host='127.0.0.1',
    port=3307,
    user='dewsweeper',
    password='dewsweeper_password123',
    database='dewsweeper3'
)

print("=" * 80)
print("SUMO DATABASE EXPLORATION")
print("=" * 80)

# 1. Check bout counts
query = """
SELECT COUNT(*) as total_bouts,
       MIN(basho_id) as earliest_basho,
       MAX(basho_id) as latest_basho
FROM boi_ozumobout
"""
df = pd.read_sql(query, conn)
print("\n1. BOUT STATISTICS:")
print(df.to_string(index=False))

# 2. Check kimarite to filter out
query = """
SELECT k.name, COUNT(*) as count
FROM boi_ozumobout b
JOIN boi_kimarite k ON b.kimarite_id = k.id
WHERE k.name LIKE '%hansoku%' OR k.name LIKE '%default%' OR k.name LIKE '%fusen%'
GROUP BY k.name
"""
df = pd.read_sql(query, conn)
print("\n2. BOUTS TO EXCLUDE (hansoku/default/fusen):")
print(df.to_string(index=False))

# 3. Check total valid bouts
query = """
SELECT COUNT(*) as valid_bouts
FROM boi_ozumobout b
JOIN boi_kimarite k ON b.kimarite_id = k.id
WHERE k.name NOT LIKE '%hansoku%'
  AND k.name NOT LIKE '%default%'
  AND k.name NOT LIKE '%fusen%'
"""
df = pd.read_sql(query, conn)
print("\n3. VALID BOUTS FOR TRAINING:")
print(df.to_string(index=False))

# 4. Sample bouts with all details
query = """
SELECT
    b.id,
    b.basho_id,
    b.day,
    wr.real_name as winner_name,
    b.winning_rikishi_rank as winner_rank,
    lr.real_name as loser_name,
    b.losing_rikishi_rank as loser_rank,
    k.name as kimarite,
    b.value
FROM boi_ozumobout b
JOIN boi_rikishi wr ON b.winning_rikishi_id = wr.id
JOIN boi_rikishi lr ON b.losing_rikishi_id = lr.id
JOIN boi_kimarite k ON b.kimarite_id = k.id
WHERE k.name NOT LIKE '%hansoku%'
  AND k.name NOT LIKE '%default%'
  AND k.name NOT LIKE '%fusen%'
ORDER BY b.basho_id DESC, b.day DESC
LIMIT 10
"""
df = pd.read_sql(query, conn)
print("\n4. SAMPLE BOUTS (10 most recent):")
print(df.to_string(index=False))

# 5. Rikishi count
query = "SELECT COUNT(*) as total_rikishi FROM boi_rikishi"
df = pd.read_sql(query, conn)
print("\n5. TOTAL WRESTLERS:")
print(df.to_string(index=False))

# 6. Check rank distribution
query = """
SELECT
    CASE
        WHEN winning_rikishi_rank = -3 THEN 'Yokozuna'
        WHEN winning_rikishi_rank = -2 THEN 'Ozeki'
        WHEN winning_rikishi_rank = -1 THEN 'Sekiwake'
        WHEN winning_rikishi_rank = 0 THEN 'Komusubi'
        WHEN winning_rikishi_rank > 0 THEN CONCAT('Maegashira-', winning_rikishi_rank)
    END as rank_group,
    COUNT(*) as bout_count
FROM boi_ozumobout b
JOIN boi_kimarite k ON b.kimarite_id = k.id
WHERE k.name NOT LIKE '%hansoku%'
  AND k.name NOT LIKE '%default%'
  AND k.name NOT LIKE '%fusen%'
GROUP BY rank_group
ORDER BY winning_rikishi_rank
LIMIT 25
"""
df = pd.read_sql(query, conn)
print("\n6. RANK DISTRIBUTION (Winners):")
print(df.to_string(index=False))

# 7. Check basho structure (how many days per basho)
query = """
SELECT basho_id, COUNT(DISTINCT day) as days_in_basho, COUNT(*) as bouts
FROM boi_ozumobout b
JOIN boi_kimarite k ON b.kimarite_id = k.id
WHERE k.name NOT LIKE '%hansoku%'
  AND k.name NOT LIKE '%default%'
  AND k.name NOT LIKE '%fusen%'
GROUP BY basho_id
ORDER BY basho_id DESC
LIMIT 10
"""
df = pd.read_sql(query, conn)
print("\n7. RECENT BASHO STRUCTURE:")
print(df.to_string(index=False))

# 8. Check if banzuke data is available
query = """
SELECT COUNT(*) as banzuke_entries,
       MIN(basho_id) as earliest_basho,
       MAX(basho_id) as latest_basho
FROM boi_ozumobanzukeentry
"""
df = pd.read_sql(query, conn)
print("\n8. BANZUKE (RANKING) DATA:")
print(df.to_string(index=False))

conn.close()
print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
