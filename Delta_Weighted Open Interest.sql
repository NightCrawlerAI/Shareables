with onetime_months
as (select month_code, start_date, peak_hours, offpeak_hours
       from blotter_trade_hours bth
      where bth.month_code not like 'Cal%'
    ), onetime_cals
as (select month_code, start_date, peak_hours, offpeak_hours
       from blotter_trade_hours bth
      where bth.month_code like 'Cal%'
    ), all_months
as (select oc.start_date start_date_cal, om.start_date start_date_month
      from onetime_months om
      inner join onetime_cals oc
         on year(om.start_date) = year(oc.start_date)
    ) 
	,base_table as (select * from ice_option_activity 
       where 
          contract in (select case when molecule_product_code like '%.O' then replace(molecule_product_code, '.O', '') else molecule_product_code end 
                            from e360_product_groups)
     )
	 , joined_table_pre 
as  (select bt.trade_date, bt.hub, bt.option_strip, bt.contract, contract_type, coalesce(bt.strike_price, 0) strike_price, 
			coalesce(total_volume, 0) total_volume, open_interest, coalesce(delta_factor, 1) delta_factor 
        from base_table bt left join ice_option_prices iop 
          on bt.trade_date = iop.trade_date 
          and bt.option_strip = iop.option_strip 
          and bt.contract = iop.contract 
          and bt.contract_type = iop.put_or_call 
          and bt.strike_price = iop.strike_price        
    )
	, base_table_onetimers_pre 
  as (select * from joined_table_pre
       where 
          contract in ('P1X','POX'))
	, base_table_non_onetimers 
  as (select * from joined_table_pre
       where 
          contract not in ('P1X','POX'))
	, base_table_onetimers
as (select trade_date, hub, start_date_month option_strip, contract, contract_type, strike_price, total_volume, open_interest
          , delta_factor
      from base_table_onetimers_pre bt
      inner join all_months am
         on bt.option_strip = am.start_date_cal
    ) 
	, joined_table_pre2 
as (select * from base_table_non_onetimers union all select * from base_table_onetimers)
   , joined_table_post 
as (select trade_date, hub, option_strip, 
		case 
			when contract in (select distinct(
								case 
									when primary_product_code like '%.O' 
								then replace(primary_product_code, '.O', '') 
								else primary_product_code end) primary_product_code
							from position_snapshot_v3
							where 
							deal_type = 'option'
							and primary_product_code not in ('LN', 'PHE'))
				and contract_type != 'F'
			then concat(contract, '.O') else contract end contract,
		contract_type, strike_price, total_volume, open_interest, delta_factor, abs(open_interest * coalesce(delta_factor, 1)) delta_weighted_oi 
		from joined_table_pre2)
		, joined_table 
as (select jt.*, sum(ps.contracts) e360_nominal_delta, ps.contract_month, ps.as_of_date, ps.primary_product_code
		from joined_table_post jt left join position_snapshot_v3 ps on jt.trade_date = ps.as_of_date and jt.option_strip = ps.contract_month
		and jt.contract = ps.primary_product_code and jt.strike_price = ps.strike_price and 
		case 
			when jt.contract_type = 'F' then 'future'
			when jt.contract_type = 'P' then 'put'
			when jt.contract_type = 'C' then 'call'
		end = ps.put_call_flag
		group by jt.trade_date, jt.hub, jt.option_strip, jt.contract, jt.contract_type, jt.strike_price, jt.total_volume,
		jt.open_interest, jt.delta_factor, jt.delta_weighted_oi, ps.contract_month, ps.as_of_date, ps.primary_product_code) 
		, final_table 
as ( select trade_date, hub, option_strip, case when contract like '%.O' then replace(contract, '.O', '') else contract end contract,
			contract_type, strike_price, sum(total_volume) total_volume, sum(open_interest) open_interest, 
			sum(delta_weighted_oi) delta_oi, sum(total_volume * delta_factor) delta_vol, delta_factor,
			coalesce(sum(e360_nominal_delta), 0) e360_delta, coalesce(sum(e360_nominal_delta * delta_factor), 0) e360_dw_oi
				from joined_table 
					group by trade_date, hub, option_strip, case when contract like '%.O' then replace(contract, '.O', '') else contract end, 
					contract_type, strike_price, delta_factor) 

				select * from final_table
					where trade_date = '2023-JUL-07'
					and contract in ('P1X', 'POX')

	select trade_date, hub, option_strip, contract, contract_type, strike_price, delta_factor, sum(total_volume) total_volume, sum(open_interest) open_interest,
	sum(delta_oi) delta_oi, sum(e360_delta) e360_delta, sum(e360_dw_oi) e360_dw_oi from final_table 
		group by trade_date, hub, option_strip, contract, contract_type, strike_price, delta_factor

		SELECT DATEADD(MONTH, -3, cast(GETDATE() as date)) 
ALTER VIEW delta_weighted_oi_vw
AS
WITH onetime_months AS (
    SELECT month_code, start_date, peak_hours, offpeak_hours
    FROM blotter_trade_hours bth
    WHERE bth.month_code NOT LIKE 'Cal%'
),
onetime_cals AS (
    SELECT month_code, start_date, peak_hours, offpeak_hours
    FROM blotter_trade_hours bth
    WHERE bth.month_code LIKE 'Cal%'
),
--Create the dictionary of cal to monthly key:value pairs-- 
all_months AS (
    SELECT oc.start_date AS start_date_cal, om.start_date AS start_date_month
    FROM onetime_months om
    INNER JOIN onetime_cals oc ON YEAR(om.start_date) = YEAR(oc.start_date)
), 
--Get ICE Activity for all product codes we've ever traded--
base_table AS (
    SELECT * 
    FROM ice_option_activity 
    WHERE contract IN (
        SELECT CASE 
            WHEN molecule_product_code LIKE '%.O' THEN REPLACE(molecule_product_code, '.O', '') 
            ELSE molecule_product_code 
        END 
        FROM e360_product_groups
    )
	AND trade_date BETWEEN DATEADD(MONTH, -3, CAST(GETDATE() AS DATE)) AND CAST(GETDATE() AS DATE)
	AND option_strip <= '2027-DEC-31'
), 
--Join the ICE Activity table with the ICE Prices table to append the delta_factor column--
joined_table_pre AS (
    SELECT bt.trade_date, bt.hub, bt.option_strip, bt.contract, contract_type, COALESCE(bt.strike_price, 0) AS strike_price, 
    COALESCE(total_volume, 0) AS total_volume, open_interest, COALESCE(delta_factor, 1) AS delta_factor 
    FROM base_table bt 
    LEFT JOIN ice_option_prices iop ON bt.trade_date = iop.trade_date 
        AND bt.option_strip = iop.option_strip 
        AND bt.contract = iop.contract 
        AND bt.contract_type = iop.put_or_call 
        AND bt.strike_price = iop.strike_price        
), 
--Process and Extended Cals out to Monthly tenors
base_table_onetimers AS (
    SELECT bt.trade_date, bt.hub, start_date_month AS option_strip, bt.contract, contract_type, strike_price, total_volume, open_interest, delta_factor
    FROM joined_table_pre bt
    INNER JOIN all_months am ON bt.option_strip = am.start_date_cal
    WHERE bt.contract IN ('P1X', 'POX')
), 
--Create final base activity table to perform delta weighted operation-- 
joined_table_pre2 AS (
    SELECT * 
    FROM joined_table_pre
    WHERE contract NOT IN ('P1X', 'POX')
    UNION ALL 
    SELECT * 
    FROM base_table_onetimers
),
--Compute Delta Weighted Market OI--
joined_table_post AS (
    SELECT trade_date, hub, option_strip, 
    CASE 
        WHEN contract IN (SELECT DISTINCT 
                            CASE 
                                WHEN primary_product_code LIKE '%.O' THEN REPLACE(primary_product_code, '.O', '') 
								--WHEN primary_product_code = 'PHH' THEN 'PHE'
                                ELSE primary_product_code 
                            END 
                        FROM position_snapshot_v3 
                        WHERE as_of_date BETWEEN DATEADD(MONTH, -3, CAST(GETDATE() AS DATE)) AND CAST(GETDATE() AS DATE) AND
						deal_type = 'option' AND primary_product_code NOT IN ('LN','PHE')
                    ) 
            AND contract_type != 'F'
        THEN CONCAT(contract, '.O') 
			--CASE 
			--	WHEN contract in ('ENO', 'ERN', 'P1X', 'PMI', 'POX') AND contract_type != 'F' THEN CONCAT(contract, '.O') 
			--	ELSE contract
		WHEN contract = 'PHE' AND contract_type NOT IN ('C', 'P')
		THEN 'PHH'
        ELSE contract 
    END AS contract,
    contract_type, strike_price, total_volume, open_interest, delta_factor, ABS(open_interest * COALESCE(delta_factor, 1)) AS delta_weighted_oi 
    FROM joined_table_pre2
),
--Join e360 Positions to DW MKt OI table--
joined_table AS (
	SELECT jt.*, 
    SUM(ps.contracts) AS e360_nominal_delta, 
    ps.contract_month, 
    ps.as_of_date, 
    ps.primary_product_code
    FROM joined_table_post jt 
    LEFT JOIN (
        SELECT *,
        CASE -- Transform jt.contract_type to match ps.put_call_flag
            WHEN put_call_flag = 'future' THEN 'F' 
            WHEN put_call_flag = 'put' THEN 'P'
            WHEN put_call_flag = 'call' THEN 'C'
        END AS put_call_flag2
        FROM position_snapshot_v3
		WHERE as_of_date BETWEEN DATEADD(MONTH, -3, CAST(GETDATE() AS DATE)) AND CAST(GETDATE() AS DATE)
    ) ps
    ON jt.trade_date = ps.as_of_date 
        AND jt.option_strip = ps.contract_month 
        AND jt.contract = ps.primary_product_code 
        AND jt.strike_price = ps.strike_price 
        AND jt.contract_type = ps.put_call_flag2
    GROUP BY jt.trade_date, jt.hub, jt.option_strip, jt.contract, jt.contract_type, jt.strike_price, jt.total_volume,
    jt.open_interest, jt.delta_factor, jt.delta_weighted_oi, ps.contract_month, ps.as_of_date, ps.primary_product_code
) ,
--Compute e360 Delta Weighted OI-- 
final_table AS (
    SELECT trade_date, hub, option_strip, 
    CASE 
        WHEN contract LIKE '%.O' THEN REPLACE(contract, '.O', '') 
        ELSE contract 
    END AS contract,
    contract_type, strike_price, 
    SUM(total_volume) AS total_volume, 
    SUM(open_interest) AS open_interest, 
    SUM(delta_weighted_oi) AS delta_oi, 
    SUM(total_volume * delta_factor) AS delta_vol, 
    delta_factor,
    COALESCE(SUM(e360_nominal_delta), 0) AS e360_delta, 
    COALESCE(SUM(e360_nominal_delta * delta_factor), 0) AS e360_dw_oi
    FROM joined_table 
    GROUP BY trade_date, hub, option_strip, 
    CASE 
        WHEN contract LIKE '%.O' THEN REPLACE(contract, '.O', '') 
        ELSE contract 
    END, 
    contract_type, strike_price, delta_factor
), 
--View Temporary Final Table--
temp_table AS (
	SELECT trade_date, contract, option_strip, 
		   SUM(total_volume) AS total_volume, 
		   SUM(open_interest) AS open_interest, 
		   SUM(delta_oi) AS delta_oi, 
		   SUM(delta_vol) AS delta_vol, 
		   AVG(delta_factor) AS delta_factor, 
		   SUM(e360_delta) AS e360_delta,
		   SUM(e360_dw_oi) AS e360_dw_oi 
	FROM final_table
	WHERE trade_date >= DATEADD(day, -30, CAST(GETDATE() AS DATE)) AND trade_date <= CAST(GETDATE() AS DATE)
	GROUP BY trade_date, contract, option_strip 
),
--Join Product Group and Subgroup to the table
map AS (
	SELECT DISTINCT
		CASE 
			WHEN molecule_product_code LIKE '%.O' 
			THEN REPLACE(molecule_product_code, '.O', '')
			ELSE molecule_product_code
		END AS molecule_product_code,
		e360_product_group,
		e360_product_subgroup
	FROM e360_product_groups
	),
final AS (
	SELECT tt.*, m.e360_product_group, m.e360_product_subgroup from temp_table tt
	LEFT JOIN map m
		ON tt.contract = m.molecule_product_code
	) 
--View Final Results--
SELECT * FROM final
where trade_date = '07-11-2023'
	
--ORDER BY 1, 2, 3;

--SELECT * FROM final_table
--WHERE trade_date BETWEEN DATEADD(MONTH, -3, CAST(GETDATE() AS DATE)) AND CAST(GETDATE() AS DATE)
SELECT * FROM final_table
WHERE trade_date = '2023-JUL-07' --used for testing
and contract in ('P1X', 'POX') --used for testing
select trade_date, contract, option_strip, sum(total_volume) total_volume, sum(open_interest) open_interest, 
	   sum(delta_oi) delta_oi, sum(delta_vol) delta_vol, avg(delta_factor) delta_factor, sum(e360_delta) e360_delta,
	   sum(e360_dw_oi) e360_dw_oi from final_table
	where trade_date >= dateadd(day, -30, cast(getdate() as date)) and trade_date <= cast(getdate() as date)
	group by trade_date, contract, option_strip 
	order by 1, 2, 3





WITH onetime_months AS (
    SELECT month_code, start_date, peak_hours, offpeak_hours
    FROM blotter_trade_hours bth
    WHERE bth.month_code NOT LIKE 'Cal%'
),
onetime_cals AS (
    SELECT month_code, start_date, peak_hours, offpeak_hours
    FROM blotter_trade_hours bth
    WHERE bth.month_code LIKE 'Cal%'
),
--Create the dictionary of cal to monthly key:value pairs-- 
all_months AS (
    SELECT oc.start_date AS start_date_cal, om.start_date AS start_date_month
    FROM onetime_months om
    INNER JOIN onetime_cals oc ON YEAR(om.start_date) = YEAR(oc.start_date)
), 
--Get ICE Activity for all product codes we've ever traded--
base_table AS (
    SELECT * 
    FROM ice_option_activity 
    WHERE contract IN (
        SELECT CASE 
            WHEN molecule_product_code LIKE '%.O' THEN REPLACE(molecule_product_code, '.O', '') 
            ELSE molecule_product_code 
        END 
        FROM e360_product_groups
    )
	AND trade_date BETWEEN DATEADD(MONTH, -3, CAST(GETDATE() AS DATE)) AND CAST(GETDATE() AS DATE)
), 
--Join the ICE Activity table with the ICE Prices table to append the delta_factor column--
joined_table_pre AS (
    SELECT bt.trade_date, bt.hub, bt.option_strip, bt.contract, contract_type, COALESCE(bt.strike_price, 0) AS strike_price, 
    COALESCE(total_volume, 0) AS total_volume, open_interest, COALESCE(delta_factor, 1) AS delta_factor 
    FROM base_table bt 
    INNER JOIN ice_option_prices iop ON bt.trade_date = iop.trade_date 
        AND bt.option_strip = iop.option_strip 
        AND bt.contract = iop.contract 
        AND bt.contract_type = iop.put_or_call 
        AND bt.strike_price = iop.strike_price        
), 
--Process and Extended Cals out to Monthly tenors
base_table_onetimers AS (
    SELECT bt.trade_date, bt.hub, start_date_month AS option_strip, bt.contract, contract_type, strike_price, total_volume, open_interest, delta_factor
    FROM joined_table_pre bt
    INNER JOIN all_months am ON bt.option_strip = am.start_date_cal
    WHERE bt.contract IN ('P1X', 'POX')
), 
--Create final base activity table to perform delta weighted operation-- 
joined_table_pre2 AS (
    SELECT * 
    FROM joined_table_pre
    WHERE contract NOT IN ('P1X', 'POX')
    UNION ALL 
    SELECT * 
    FROM base_table_onetimers
),
--Compute Delta Weighted Market OI--
joined_table_post AS (
    SELECT trade_date, hub, option_strip, 
    CASE 
        WHEN contract IN (SELECT DISTINCT 
                            CASE 
                                WHEN primary_product_code LIKE '%.O' THEN REPLACE(primary_product_code, '.O', '') 
                                ELSE primary_product_code 
                            END 
                        FROM position_snapshot_v3 
                        WHERE as_of_date BETWEEN DATEADD(MONTH, -3, CAST(GETDATE() AS DATE)) AND CAST(GETDATE() AS DATE) AND
						deal_type = 'option' AND primary_product_code NOT IN ('LN', 'PHE')
                    ) 
            AND contract_type != 'F' 
        THEN CONCAT(contract, '.O') 
        ELSE contract 
    END AS contract,
    contract_type, strike_price, total_volume, open_interest, delta_factor, ABS(open_interest * COALESCE(delta_factor, 1)) AS delta_weighted_oi 
    FROM joined_table_pre2
),
--Join e360 Positions to DW MKt OI table--
joined_table AS (
    SELECT jt.*, 
    SUM(ps.contracts) AS e360_nominal_delta, 
    ps.contract_month, 
    ps.as_of_date, 
    ps.primary_product_code
    FROM joined_table_post jt 
    LEFT JOIN (
        SELECT *,
        CASE -- Transform jt.contract_type to match ps.put_call_flag
            WHEN put_call_flag = 'future' THEN 'F' 
            WHEN put_call_flag = 'put' THEN 'P'
            WHEN put_call_flag = 'call' THEN 'C'
        END AS put_call_flag2
        FROM position_snapshot_v3
        WHERE as_of_date BETWEEN DATEADD(MONTH, -3, CAST(GETDATE() AS DATE)) AND CAST(GETDATE() AS DATE)
    ) ps
    ON jt.trade_date = ps.as_of_date 
        AND jt.option_strip = ps.contract_month 
        AND jt.contract = ps.primary_product_code 
        AND jt.strike_price = ps.strike_price 
        AND jt.contract_type = ps.put_call_flag2
    GROUP BY jt.trade_date, jt.hub, jt.option_strip, jt.contract, jt.contract_type, jt.strike_price, jt.total_volume,
    jt.open_interest, jt.delta_factor, jt.delta_weighted_oi, ps.contract_month, ps.as_of_date, ps.primary_product_code
),
--Compute e360 Delta Weighted OI-- 
final_table AS (
    SELECT trade_date, hub, option_strip, 
    CASE 
        WHEN contract LIKE '%.O' THEN REPLACE(contract, '.O', '') 
        ELSE contract 
    END AS contract,
    contract_type, strike_price, 
    SUM(total_volume) AS total_volume, 
    SUM(open_interest) AS open_interest, 
    SUM(delta_weighted_oi) AS delta_oi, 
    SUM(total_volume * delta_factor) AS delta_vol, 
    delta_factor,
    COALESCE(SUM(e360_nominal_delta), 0) AS e360_delta, 
    COALESCE(SUM(e360_nominal_delta * delta_factor), 0) AS e360_dw_oi
    FROM joined_table 
    GROUP BY trade_date, hub, option_strip, 
    CASE 
        WHEN contract LIKE '%.O' THEN REPLACE(contract, '.O', '') 
        ELSE contract 
    END, 
    contract_type, strike_price, delta_factor
) 
--View Final Results--
SELECT min(trade_date) min_date, max(trade_date) max_date FROM final_table
WHERE trade_date = '2023-JUL-07' --used for testing
and contract in ('P1X', 'POX') --used for testing








-- Index on ice_option_activity table
CREATE INDEX idx_ice_option_activity_trade_date_contract
ON ice_option_activity (trade_date, contract, option_strip, contract_type, strike_price);

-- Index on ice_option_prices table
CREATE INDEX idx_ice_option_prices_trade_date_contract
ON ice_option_prices (trade_date, contract, option_strip, put_or_call, strike_price);

-- Index on e360_product_groups table
CREATE INDEX idx_e360_product_groups_molecule_product_code
ON e360_product_groups (molecule_product_code);

-- Index on position_snapshot_v3 table
CREATE INDEX idx_position_snapshot_v3_as_of_date_contract
ON position_snapshot_v3 (as_of_date, primary_product_code, contract_month, strike_price, put_call_flag);

