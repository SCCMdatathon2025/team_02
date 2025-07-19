-- SELECT DISTINCT po.procedure_concept_id, c.concept_name
-- FROM `sccm-discovery.rediscover_datathon_2025.procedure_occurrence` po
-- JOIN `sccm-discovery.rediscover_datathon_2025.concept` c
--   ON po.procedure_concept_id = c.concept_id
-- WHERE LOWER(c.concept_name) LIKE '%consult%'
-- LIMIT 200;


-- SELECT COUNT(*) as total_count
-- FROM `sccm-discovery.rediscover_datathon_2025.care_site`
-- LIMIT 100

-- Device exposure - get ventilator
-- SELECT DISTINCT de.device_concept_id, c.concept_name
-- FROM `sccm-discovery.rediscover_datathon_2025.device_exposure` de
-- JOIN `sccm-discovery.rediscover_datathon_2025.concept` c
--   ON de.device_concept_id = c.concept_id
-- WHERE LOWER(c.concept_name) LIKE '%ventilator%'

-- Device exposure - o2 supply

-- SELECT DISTINCT de.device_concept_id, c.concept_name
-- FROM `sccm-discovery.rediscover_datathon_2025.device_exposure` de
-- JOIN `sccm-discovery.rediscover_datathon_2025.concept` c
--   ON de.device_concept_id = c.concept_id
-- WHERE LOWER(c.concept_name) LIKE '%oxygen%'

-- Analyse LOS from visit_occurrence

-- SELECT 
--   visit_occurrence_id,
--   person_id,
--   visit_start_date,
--   visit_end_date,
--   date_diff(visit_end_date, visit_start_date, day) as los
-- FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence`


-- Mortality Analysis
-- See when they died, how long after their LATEST discharge
-- Only considers the most recent visit for each person
-- Filters for patients aged >= 18 and removes those who died before their visit
-- Ensures each person_id appears only once by handling potential duplicate death records

-- WITH latest_visits AS (
--   SELECT 
--     vo.person_id,
--     vo.visit_occurrence_id,
--     vo.visit_start_datetime,
--     vo.visit_end_datetime,
--     ROW_NUMBER() OVER (PARTITION BY vo.person_id ORDER BY vo.visit_end_datetime DESC) as rn
--   FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence` vo
--   JOIN `sccm-discovery.rediscover_datathon_2025.person` p
--     ON vo.person_id = p.person_id
--   WHERE EXTRACT(YEAR FROM vo.visit_start_datetime) - p.year_of_birth >= 18  -- Age >= 18 at visit
-- ),
-- latest_deaths AS (
--   SELECT 
--     person_id,
--     death_datetime,
--     ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY death_datetime DESC) as death_rn
--   FROM `sccm-discovery.rediscover_datathon_2025.death`
--   WHERE death_datetime IS NOT NULL
-- )
-- SELECT 
--   lv.person_id,
--   lv.visit_occurrence_id,
--   lv.visit_start_datetime,
--   lv.visit_end_datetime as latest_discharge_datetime,
--   ld.death_datetime,
--   datetime_diff(ld.death_datetime, lv.visit_end_datetime, day) as days_to_death,
--   CASE 
--     WHEN ld.death_datetime >= lv.visit_end_datetime THEN 'Death after discharge'
--     WHEN ld.death_datetime BETWEEN lv.visit_start_datetime AND lv.visit_end_datetime THEN 'Death during visit'
--   END as death_timing
-- FROM latest_visits lv
-- JOIN latest_deaths ld
--   ON lv.person_id = ld.person_id
-- WHERE lv.rn = 1  -- Only the latest visit per person
--   AND ld.death_rn = 1  -- Only the latest death record per person
--   -- AND ld.death_datetime >= lv.visit_end_datetime  -- Remove patients who died before their visit
-- ORDER BY days_to_death DESC


-- SELECT visit_type_concept_id FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence` a INNER JOIN `sccm-discovery.rediscover_datathon_2025.concept` b by a.visit_type_concept_id

-- Optimized query to get unique visit type concept names
-- SELECT DISTINCT b.concept_name 
-- FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence` a 
-- INNER JOIN `sccm-discovery.rediscover_datathon_2025.concept` b 
--   ON a.visit_concept_id = b.concept_id
-- WHERE b.concept_name IS NOT NULL
-- ORDER BY b.concept_name

-- SELECT DISTINCT visit_concept_id FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence`





-- SELECT p.race_concept_id, c.concept_name, COUNT(*) as count 
-- FROM `sccm-discovery.rediscover_datathon_2025.person` p
-- INNER JOIN `sccm-discovery.rediscover_datathon_2025.concept` c 
--   ON p.race_concept_id = c.concept_id 
-- WHERE c.concept_name IS NOT NULL
-- GROUP BY p.race_concept_id, c.concept_name


-- Count total visit occurrences and distinct persons where they are 18 or older
SELECT 
  DISTINCT vo.person_id
FROM `sccm-discovery.rediscover_datathon_2025.visit_occurrence` vo
JOIN `sccm-discovery.rediscover_datathon_2025.person` p
  ON vo.person_id = p.person_id
WHERE EXTRACT(YEAR FROM vo.visit_start_datetime) - p.year_of_birth >= 18

-- SELECT COUNT(*) as count 
-- FROM `sccm-discovery.rediscover_datathon_2025.person`