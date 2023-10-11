#!/usr/bin/python3

queries_o1 = {
    # Q 39a
    "39a": """WITH inv AS (SELECT warehouse_name, w_warehouse_sk, i_item_sk, d_moy, stdev, mean, CASE mean WHEN 0 THEN NULL ELSE stdev/mean END cov FROM (SELECT MIN(w_warehouse_name) warehouse_name, w_warehouse_sk, i_item_sk, d_moy, stddev_samp(inv_quantity_on_hand) stdev, avg(inv_quantity_on_hand) mean FROM inventory, item, warehouse, date_dim WHERE inv_item_sk = i_item_sk AND inv_warehouse_sk = w_warehouse_sk AND inv_date_sk = d_date_sk AND d_year =2001 GROUP BY w_warehouse_sk, i_item_sk, d_moy) foo WHERE CASE mean WHEN 0 THEN 0 ELSE stdev/mean END > 1) SELECT inv1.w_warehouse_sk, inv1.i_item_sk, inv1.d_moy, inv1.mean, inv1.cov, inv2.w_warehouse_sk, inv2.i_item_sk, inv2.d_moy, inv2.mean, inv2.cov FROM inv inv1, inv inv2 WHERE inv1.i_item_sk = inv2.i_item_sk AND inv1.w_warehouse_sk = inv2.w_warehouse_sk AND inv1.d_moy=1 AND inv2.d_moy=1+1 ORDER BY inv1.w_warehouse_sk, inv1.i_item_sk, inv1.d_moy, inv1.mean, inv1.cov, inv2.d_moy, inv2.mean, inv2.cov;""",
    # Q 39b
    "39b": """WITH inv AS (SELECT warehouse_name, w_warehouse_sk, i_item_sk, d_moy, stdev, mean, CASE mean WHEN 0 THEN NULL ELSE stdev/mean END cov FROM (SELECT min(w_warehouse_name) warehouse_name, w_warehouse_sk, i_item_sk, d_moy, stddev_samp(inv_quantity_on_hand) stdev, avg(inv_quantity_on_hand) mean FROM inventory, item, warehouse, date_dim WHERE inv_item_sk = i_item_sk AND inv_warehouse_sk = w_warehouse_sk AND inv_date_sk = d_date_sk AND d_year =2001 GROUP BY w_warehouse_sk, i_item_sk, d_moy) foo WHERE CASE mean WHEN 0 THEN 0 ELSE stdev/mean END > 1) SELECT inv1.w_warehouse_sk, inv1.i_item_sk, inv1.d_moy, inv1.mean, inv1.cov, inv2.w_warehouse_sk, inv2.i_item_sk, inv2.d_moy, inv2.mean, inv2.cov FROM inv inv1, inv inv2 WHERE inv1.i_item_sk = inv2.i_item_sk AND inv1.w_warehouse_sk = inv2.w_warehouse_sk AND inv1.d_moy=1 AND inv2.d_moy=1+1 AND inv1.cov > 1.5 ORDER BY inv1.w_warehouse_sk, inv1.i_item_sk, inv1.d_moy, inv1.mean, inv1.cov, inv2.d_moy, inv2.mean, inv2.cov;"""
}

queries_o3 = {
    # Q01: fetch min(d_date_sk), max(d_date_sk) for d_year = 2000 (-> 2451545, 2451910)
    "01": """select min(d_date_sk), max(d_date_sk) from date_dim where d_year = 2000; WITH customer_total_return AS (SELECT sr_customer_sk AS ctr_customer_sk, sr_store_sk AS ctr_store_sk, sum(sr_return_amt) AS ctr_total_return FROM store_returns WHERE sr_returned_date_sk BETWEEN 2451545 AND 2451910 GROUP BY sr_customer_sk, sr_store_sk) SELECT c_customer_id FROM customer_total_return ctr1, store, customer WHERE ctr1.ctr_total_return > (SELECT avg(ctr_total_return)*1.2 FROM customer_total_return ctr2 WHERE ctr1.ctr_store_sk = ctr2.ctr_store_sk) AND s_store_sk = ctr1.ctr_store_sk AND s_state = 'TN' AND ctr1.ctr_customer_sk = c_customer_sk ORDER BY c_customer_id LIMIT 100;""",
    # Q07: fetch min(d_date_sk), max(d_date_sk) for d_year = 2000 (-> 2451545, 2451910)
    "07": """select min(d_date_sk), max(d_date_sk) from date_dim where d_year = 2000; SELECT i_item_id, avg(ss_quantity) agg1, avg(ss_list_price) agg2, avg(ss_coupon_amt) agg3, avg(ss_sales_price) agg4 FROM store_sales, customer_demographics, item, promotion WHERE ss_sold_date_sk BETWEEN 2451545 AND 2451910 AND ss_item_sk = i_item_sk AND ss_cdemo_sk = cd_demo_sk AND ss_promo_sk = p_promo_sk AND cd_gender = 'M' AND cd_marital_status = 'S' AND cd_education_status = 'College' AND (p_channel_email = 'N' OR p_channel_event = 'N') GROUP BY i_item_id ORDER BY i_item_id LIMIT 100;""",
    # Q13: fetch min(d_date_sk), max(d_date_sk) for d_year = 2001 (-> 2451911, 2452275)
    "13": """select min(d_date_sk), max(d_date_sk) from date_dim where d_year = 2001; SELECT avg(ss_quantity) , avg(ss_ext_sales_price) , avg(ss_ext_wholesale_cost) , sum(ss_ext_wholesale_cost) FROM store_sales , store , customer_demographics , household_demographics , customer_address , date_dim WHERE s_store_sk = ss_store_sk AND ss_sold_date_sk BETWEEN 2451911 AND 2452275 and((ss_hdemo_sk=hd_demo_sk AND cd_demo_sk = ss_cdemo_sk AND cd_marital_status = 'M' AND cd_education_status = 'Advanced Degree' AND ss_sales_price BETWEEN 100.00 AND 150.00 AND hd_dep_count = 3) OR (ss_hdemo_sk=hd_demo_sk AND cd_demo_sk = ss_cdemo_sk AND cd_marital_status = 'S' AND cd_education_status = 'College' AND ss_sales_price BETWEEN 50.00 AND 100.00 AND hd_dep_count = 1 ) OR (ss_hdemo_sk=hd_demo_sk AND cd_demo_sk = ss_cdemo_sk AND cd_marital_status = 'W' AND cd_education_status = '2 yr Degree' AND ss_sales_price BETWEEN 150.00 AND 200.00 AND hd_dep_count = 1)) and((ss_addr_sk = ca_address_sk AND ca_country = 'United States' AND ca_state IN ('TX', 'OH', 'TX') AND ss_net_profit BETWEEN 100 AND 200) OR (ss_addr_sk = ca_address_sk AND ca_country = 'United States' AND ca_state IN ('OR', 'NM', 'KY') AND ss_net_profit BETWEEN 150 AND 300) OR (ss_addr_sk = ca_address_sk AND ca_country = 'United States' AND ca_state IN ('VA', 'TX', 'MS') AND ss_net_profit BETWEEN 50 AND 250));""",
    # Q16: fetch min(d_date_sk), max(d_date_sk) for d_date BETWEEN '2002-02-01' AND cast('2002-04-02' AS date) (-> 2452307, 2452367)
    "16": """select min(d_date_sk), max(d_date_sk) from date_dim where d_date BETWEEN '2002-02-01' AND cast('2002-04-02' AS date); SELECT count(DISTINCT cs_order_number) AS "order count", sum(cs_ext_ship_cost) AS "total shipping cost", sum(cs_net_profit) AS "total net profit" FROM catalog_sales cs1, customer_address, call_center WHERE cs1.cs_ship_date_sk BETWEEN 2452307 AND 2452367 AND cs1.cs_ship_addr_sk = ca_address_sk AND ca_state = 'GA' AND cs1.cs_call_center_sk = cc_call_center_sk AND cc_county = 'Williamson County' AND EXISTS (SELECT * FROM catalog_sales cs2 WHERE cs1.cs_order_number = cs2.cs_order_number AND cs1.cs_warehouse_sk <> cs2.cs_warehouse_sk) AND NOT EXISTS (SELECT * FROM catalog_returns cr1 WHERE cs1.cs_order_number = cr1.cr_order_number) ORDER BY count(DISTINCT cs_order_number) LIMIT 100;""",
    # Q17: fetch min(d_date_sk), max(d_date_sk) for d_quarter_name = '2001Q1' (-> 2451911, 2452001)
    "17": """select min(d_date_sk), max(d_date_sk) from date_dim where d_quarter_name = '2001Q1'; SELECT i_item_id, i_item_desc, s_state, count(ss_quantity) AS store_sales_quantitycount, avg(ss_quantity) AS store_sales_quantityave, stddev_samp(ss_quantity) AS store_sales_quantitystdev, stddev_samp(ss_quantity)/avg(ss_quantity) AS store_sales_quantitycov, count(sr_return_quantity) AS store_returns_quantitycount, avg(sr_return_quantity) AS store_returns_quantityave, stddev_samp(sr_return_quantity) AS store_returns_quantitystdev, stddev_samp(sr_return_quantity)/avg(sr_return_quantity) AS store_returns_quantitycov, count(cs_quantity) AS catalog_sales_quantitycount, avg(cs_quantity) AS catalog_sales_quantityave, stddev_samp(cs_quantity) AS catalog_sales_quantitystdev, stddev_samp(cs_quantity)/avg(cs_quantity) AS catalog_sales_quantitycov FROM store_sales, store_returns, catalog_sales, date_dim d2, date_dim d3, store, item WHERE ss_sold_date_sk BETWEEN 2451911 AND 2452001 AND i_item_sk = ss_item_sk AND s_store_sk = ss_store_sk AND ss_customer_sk = sr_customer_sk AND ss_item_sk = sr_item_sk AND ss_ticket_number = sr_ticket_number AND sr_returned_date_sk = d2.d_date_sk AND d2.d_quarter_name IN ('2001Q1', '2001Q2', '2001Q3') AND sr_customer_sk = cs_bill_customer_sk AND sr_item_sk = cs_item_sk AND cs_sold_date_sk = d3.d_date_sk AND d3.d_quarter_name IN ('2001Q1', '2001Q2', '2001Q3') GROUP BY i_item_id, i_item_desc, s_state ORDER BY i_item_id, i_item_desc, s_state LIMIT 100;""",
    # Q26: fetch min(d_date_sk), max(d_date_sk) for d_year = 2000 (-> 2451545, 2451910)
    "26": """select min(d_date_sk), max(d_date_sk) from date_dim where d_year = 2000; SELECT i_item_id, avg(cs_quantity) agg1, avg(cs_list_price) agg2, avg(cs_coupon_amt) agg3, avg(cs_sales_price) agg4 FROM catalog_sales, customer_demographics, item, promotion WHERE cs_sold_date_sk BETWEEN 2451545 AND 2451910 AND cs_item_sk = i_item_sk AND cs_bill_cdemo_sk = cd_demo_sk AND cs_promo_sk = p_promo_sk AND cd_gender = 'M' AND cd_marital_status = 'S' AND cd_education_status = 'College' AND (p_channel_email = 'N' OR p_channel_event = 'N') GROUP BY i_item_id ORDER BY i_item_id LIMIT 100;""",
    # Q32: fetch min(d_date_sk), max(d_date_sk) for d_date BETWEEN '2000-01-27' AND cast('2000-04-26' AS date) (-> 2451571, 2451661)
    "32": """select min(d_date_sk), max(d_date_sk) from date_dim where d_date BETWEEN '2000-01-27' AND cast('2000-04-26' AS date); SELECT sum(cs_ext_discount_amt) AS "excess discount amount" FROM catalog_sales , item WHERE i_manufact_id = 977 AND i_item_sk = cs_item_sk AND cs_sold_date_sk BETWEEN 2451571 AND 2451661 AND cs_ext_discount_amt > ( SELECT 1.3 * avg(cs_ext_discount_amt) FROM catalog_sales WHERE cs_item_sk = i_item_sk AND cs_sold_date_sk BETWEEN 2451571 AND 2451661) LIMIT 100;""",
    # Q37: fetch min(d_date_sk), max(d_date_sk) for d_date BETWEEN cast('2000-02-01' AS date) AND cast('2000-04-01' AS date) (-> 2451576, 2451636)
    "37": """select min(d_date_sk), max(d_date_sk) from date_dim where d_date BETWEEN cast('2000-02-01' AS date) AND cast('2000-04-01' AS date); SELECT i_item_id, i_item_desc, i_current_price FROM item, inventory, catalog_sales WHERE i_current_price BETWEEN 68 AND 68 + 30 AND inv_item_sk = i_item_sk AND  inv_date_sk BETWEEN 2451576 AND 2451636 AND i_manufact_id IN (677, 940, 694, 808) AND inv_quantity_on_hand BETWEEN 100 AND 500 AND cs_item_sk = i_item_sk GROUP BY i_item_id, i_item_desc, i_current_price ORDER BY i_item_id LIMIT 100;""",
    # Q48: fetch min(d_date_sk), max(d_date_sk) for d_year = 2000 (-> 2451545, 2451910)
    "48": """select min(d_date_sk), max(d_date_sk) from date_dim where d_year = 2000; SELECT SUM(ss_quantity) FROM store_sales, store, customer_demographics, customer_address WHERE s_store_sk = ss_store_sk AND ss_sold_date_sk BETWEEN 2451545 AND 2451910 AND ((cd_demo_sk = ss_cdemo_sk AND cd_marital_status = 'M' AND cd_education_status = '4 yr Degree' AND ss_sales_price BETWEEN 100.00 AND 150.00) OR (cd_demo_sk = ss_cdemo_sk AND cd_marital_status = 'D' AND cd_education_status = '2 yr Degree' AND ss_sales_price BETWEEN 50.00 AND 100.00) OR (cd_demo_sk = ss_cdemo_sk AND cd_marital_status = 'S' AND cd_education_status = 'College' AND ss_sales_price BETWEEN 150.00 AND 200.00)) AND ((ss_addr_sk = ca_address_sk AND ca_country = 'United States' AND ca_state IN ('CO', 'OH', 'TX') AND ss_net_profit BETWEEN 0 AND 2000) OR (ss_addr_sk = ca_address_sk AND ca_country = 'United States' AND ca_state IN ('OR', 'MN', 'KY') AND ss_net_profit BETWEEN 150 AND 3000) OR (ss_addr_sk = ca_address_sk AND ca_country = 'United States' AND ca_state IN ('VA', 'CA', 'MS') AND ss_net_profit BETWEEN 50 AND 25000));""",
    # Q62: fetch min(d_date_sk), max(d_date_sk) for d_month_seq BETWEEN 1200 AND 1200 + 11 (-> 2451545, 2451910)
    "62": """select min(d_date_sk), max(d_date_sk) from date_dim where d_month_seq BETWEEN 1200 AND 1200 + 11; SELECT w_substr, sm_type, web_name, sum(CASE WHEN (ws_ship_date_sk - ws_sold_date_sk <= 30) THEN 1 ELSE 0 END) AS "30 days", sum(CASE WHEN (ws_ship_date_sk - ws_sold_date_sk > 30) AND (ws_ship_date_sk - ws_sold_date_sk <= 60) THEN 1 ELSE 0 END) AS "31-60 days", sum(CASE WHEN (ws_ship_date_sk - ws_sold_date_sk > 60) AND (ws_ship_date_sk - ws_sold_date_sk <= 90) THEN 1 ELSE 0 END) AS "61-90 days", sum(CASE WHEN (ws_ship_date_sk - ws_sold_date_sk > 90) AND (ws_ship_date_sk - ws_sold_date_sk <= 120) THEN 1 ELSE 0 END) AS "91-120 days", sum(CASE WHEN (ws_ship_date_sk - ws_sold_date_sk > 120) THEN 1 ELSE 0 END) AS ">120 days" FROM web_sales, (SELECT SUBSTR(w_warehouse_name,1,20) w_substr, * FROM warehouse) sq1, ship_mode, web_site, WHERE ws_ship_date_sk BETWEEN 2451545 AND 2451910 AND ws_warehouse_sk = w_warehouse_sk AND ws_ship_mode_sk = sm_ship_mode_sk AND ws_web_site_sk = web_site_sk GROUP BY w_substr, sm_type, web_name ORDER BY w_substr, sm_type, web_name LIMIT 100;""",
    # Q65: fetch min(d_date_sk), max(d_date_sk) for d_month_seq BETWEEN 1176 AND 1176+11 (-> 2450815, 2451179)
    "65": """select min(d_date_sk), max(d_date_sk) from date_dim where d_month_seq BETWEEN 1176 AND 1176+11; SELECT s_store_name, i_item_desc, sc.revenue, i_current_price, i_wholesale_cost, i_brand FROM store, item,  (SELECT ss_store_sk, avg(revenue) AS ave FROM (SELECT ss_store_sk, ss_item_sk, sum(ss_sales_price) AS revenue FROM store_sales WHERE ss_sold_date_sk BETWEEN 2450815 AND 2451179 GROUP BY ss_store_sk, ss_item_sk) sa GROUP BY ss_store_sk) sb,  (SELECT ss_store_sk, ss_item_sk, sum(ss_sales_price) AS revenue FROM store_sales WHERE ss_sold_date_sk BETWEEN 2450815 AND 2451179 GROUP BY ss_store_sk, ss_item_sk) sc WHERE sb.ss_store_sk = sc.ss_store_sk AND sc.revenue <= 0.1 * sb.ave AND s_store_sk = sc.ss_store_sk AND i_item_sk = sc.ss_item_sk ORDER BY s_store_name, i_item_desc LIMIT 100;""",
    # Q81: fetch min(d_date_sk), max(d_date_sk) for d_year = 2000 (-> 2451545, 2451910)
    "81": """select min(d_date_sk), max(d_date_sk) from date_dim where d_year = 2000; WITH customer_total_return AS (SELECT cr_returning_customer_sk AS ctr_customer_sk , ca_state AS ctr_state, sum(cr_return_amt_inc_tax) AS ctr_total_return FROM catalog_returns , customer_address WHERE cr_returned_date_sk BETWEEN 2451545 AND 2451910 AND cr_returning_addr_sk = ca_address_sk GROUP BY cr_returning_customer_sk , ca_state) SELECT c_customer_id, c_salutation, c_first_name, c_last_name, ca_street_number, ca_street_name , ca_street_type, ca_suite_number, ca_city, ca_county, ca_state, ca_zip, ca_country, ca_gmt_offset , ca_location_type, ctr_total_return FROM customer_total_return ctr1 , customer_address , customer WHERE ctr1.ctr_total_return > (SELECT avg(ctr_total_return)*1.2 FROM customer_total_return ctr2 WHERE ctr1.ctr_state = ctr2.ctr_state) AND ca_address_sk = c_current_addr_sk AND ca_state = 'GA' AND ctr1.ctr_customer_sk = c_customer_sk ORDER BY c_customer_id, c_salutation, c_first_name, c_last_name, ca_street_number, ca_street_name , ca_street_type, ca_suite_number, ca_city, ca_county, ca_state, ca_zip, ca_country, ca_gmt_offset , ca_location_type, ctr_total_return LIMIT 100;""",
    # Q82: fetch min(d_date_sk), max(d_date_sk) for d_date BETWEEN cast('2000-05-25' AS date) AND cast('2000-07-24' AS date) (-> 2451690, 2451750)
    "82": """select min(d_date_sk), max(d_date_sk) from date_dim where d_date BETWEEN cast('2000-05-25' AS date) AND cast('2000-07-24' AS date); SELECT i_item_id , i_item_desc , i_current_price FROM item, inventory, store_sales WHERE i_current_price BETWEEN 62 AND 62+30 AND inv_item_sk = i_item_sk AND  inv_date_sk BETWEEN 2451690 AND 2451750 AND i_manufact_id IN (129, 270, 821, 423) AND inv_quantity_on_hand BETWEEN 100 AND 500 AND ss_item_sk = i_item_sk GROUP BY i_item_id, i_item_desc, i_current_price ORDER BY i_item_id LIMIT 100;""",
    # Q85: fetch min(d_date_sk), max(d_date_sk) for d_year = 2000 (-> 2451545, 2451910)
    "85": """select min(d_date_sk), max(d_date_sk) from date_dim where d_year = 2000; SELECT SUBSTR(r_reason_desc,1,20) , avg(ws_quantity) , avg(wr_refunded_cash) , avg(wr_fee) FROM web_sales, web_returns, web_page, customer_demographics cd1, customer_demographics cd2, customer_address, reason WHERE ws_web_page_sk = wp_web_page_sk AND ws_item_sk = wr_item_sk AND ws_order_number = wr_order_number AND ws_sold_date_sk BETWEEN 2451545 AND 2451910 AND cd1.cd_demo_sk = wr_refunded_cdemo_sk AND cd2.cd_demo_sk = wr_returning_cdemo_sk AND ca_address_sk = wr_refunded_addr_sk AND r_reason_sk = wr_reason_sk AND ( ( cd1.cd_marital_status = 'M' AND cd1.cd_marital_status = cd2.cd_marital_status AND cd1.cd_education_status = 'Advanced Degree' AND cd1.cd_education_status = cd2.cd_education_status AND ws_sales_price BETWEEN 100.00 AND 150.00 ) OR ( cd1.cd_marital_status = 'S' AND cd1.cd_marital_status = cd2.cd_marital_status AND cd1.cd_education_status = 'College' AND cd1.cd_education_status = cd2.cd_education_status AND ws_sales_price BETWEEN 50.00 AND 100.00 ) OR ( cd1.cd_marital_status = 'W' AND cd1.cd_marital_status = cd2.cd_marital_status AND cd1.cd_education_status = '2 yr Degree' AND cd1.cd_education_status = cd2.cd_education_status AND ws_sales_price BETWEEN 150.00 AND 200.00 ) ) AND ( ( ca_country = 'United States' AND ca_state IN ('IN', 'OH', 'NJ') AND ws_net_profit BETWEEN 100 AND 200) OR ( ca_country = 'United States' AND ca_state IN ('WI', 'CT', 'KY') AND ws_net_profit BETWEEN 150 AND 300) OR ( ca_country = 'United States' AND ca_state IN ('LA', 'IA', 'AR') AND ws_net_profit BETWEEN 50 AND 250) ) GROUP BY r_reason_desc ORDER BY SUBSTR(r_reason_desc,1,20) , avg(ws_quantity) , avg(wr_refunded_cash) , avg(wr_fee) LIMIT 100;""",
    # Q92: fetch min(d_date_sk), max(d_date_sk) for d_date BETWEEN '2000-01-27' AND cast('2000-04-26' AS date) (-> 2451571, 2451661)
    "92": """select min(d_date_sk), max(d_date_sk) from date_dim where d_date BETWEEN '2000-01-27' AND cast('2000-04-26' AS date); SELECT sum(ws_ext_discount_amt) AS "Excess Discount Amount" FROM web_sales, item WHERE i_manufact_id = 350 AND i_item_sk = ws_item_sk AND ws_sold_date_sk BETWEEN 2451571 AND 2451661 AND ws_ext_discount_amt > (SELECT 1.3 * avg(ws_ext_discount_amt) FROM web_sales, date_dim WHERE ws_item_sk = i_item_sk AND ws_sold_date_sk BETWEEN 2451571 AND 2451661 ) ORDER BY sum(ws_ext_discount_amt) LIMIT 100;""",
    # Q94: fetch min(d_date_sk), max(d_date_sk) for d_date BETWEEN '1999-02-01' AND cast('1999-04-02' AS date) (-> 2451211, 2451271)
    "94": """select min(d_date_sk), max(d_date_sk) from date_dim where d_date BETWEEN '1999-02-01' AND cast('1999-04-02' AS date);  SELECT count(DISTINCT ws_order_number) AS "order count" , sum(ws_ext_ship_cost) AS "total shipping cost" , sum(ws_net_profit) AS "total net profit" FROM web_sales ws1 , customer_address , web_site WHERE ws1.ws_ship_date_sk BETWEEN 2451211 AND 2451271 AND ws1.ws_ship_addr_sk = ca_address_sk AND ca_state = 'IL' AND ws1.ws_web_site_sk = web_site_sk AND web_company_name = 'pri' AND EXISTS (SELECT * FROM web_sales ws2 WHERE ws1.ws_order_number = ws2.ws_order_number AND ws1.ws_warehouse_sk <> ws2.ws_warehouse_sk) AND NOT exists (SELECT * FROM web_returns wr1 WHERE ws1.ws_order_number = wr1.wr_order_number) ORDER BY count(DISTINCT ws_order_number) LIMIT 100;""",
    # Q95: fetch min(d_date_sk), max(d_date_sk) for d_date BETWEEN '1999-02-01' AND cast('1999-04-02' AS date) (-> 2451211, 2451271)
    "95": """select min(d_date_sk), max(d_date_sk) from date_dim where d_date BETWEEN '1999-02-01' AND cast('1999-04-02' AS date); WITH ws_wh AS (SELECT ws1.ws_order_number, ws1.ws_warehouse_sk wh1, ws2.ws_warehouse_sk wh2 FROM web_sales ws1, web_sales ws2 WHERE ws1.ws_order_number = ws2.ws_order_number AND ws1.ws_warehouse_sk <> ws2.ws_warehouse_sk) SELECT count(DISTINCT ws_order_number) AS "order count" , sum(ws_ext_ship_cost) AS "total shipping cost" , sum(ws_net_profit) AS "total net profit" FROM web_sales ws1 , customer_address , web_site WHERE ws1.ws_ship_date_sk BETWEEN 2451211 AND 2451271 AND ws1.ws_ship_addr_sk = ca_address_sk AND ca_state = 'IL' AND ws1.ws_web_site_sk = web_site_sk AND web_company_name = 'pri' AND ws1.ws_order_number IN (SELECT ws_order_number FROM ws_wh) AND ws1.ws_order_number IN (SELECT wr_order_number FROM web_returns, ws_wh WHERE wr_order_number = ws_wh.ws_order_number) ORDER BY count(DISTINCT ws_order_number) LIMIT 100;""",
    # Q97: fetch min(d_date_sk), max(d_date_sk) for  d_month_seq BETWEEN 1200 AND 1200 + 11 (-> 2451545, 2451910)
    "97": """select min(d_date_sk), max(d_date_sk) from date_dim where  d_month_seq BETWEEN 1200 AND 1200 + 11; WITH ssci AS (SELECT ss_customer_sk customer_sk , ss_item_sk item_sk FROM store_sales WHERE ss_sold_date_sk BETWEEN 2451545 AND 2451910 GROUP BY ss_customer_sk , ss_item_sk), csci as ( SELECT cs_bill_customer_sk customer_sk ,cs_item_sk item_sk FROM catalog_sales WHERE cs_sold_date_sk BETWEEN 2451545 AND 2451910 GROUP BY cs_bill_customer_sk ,cs_item_sk) SELECT sum(CASE WHEN ssci.customer_sk IS NOT NULL AND csci.customer_sk IS NULL THEN 1 ELSE 0 END) store_only , sum(CASE WHEN ssci.customer_sk IS NULL AND csci.customer_sk IS NOT NULL THEN 1 ELSE 0 END) catalog_only , sum(CASE WHEN ssci.customer_sk IS NOT NULL AND csci.customer_sk IS NOT NULL THEN 1 ELSE 0 END) store_and_catalog FROM ssci FULL OUTER JOIN csci ON (ssci.customer_sk=csci.customer_sk AND ssci.item_sk = csci.item_sk) LIMIT 100;""",
    # Q99: fetch min(d_date_sk), max(d_date_sk) for  d_month_seq BETWEEN 1200 AND 1200 + 11 (-> 2451545, 2451910)
    "99": """select min(d_date_sk), max(d_date_sk) from date_dim where  d_month_seq BETWEEN 1200 AND 1200 + 11; SELECT w_substr , sm_type , cc_name , sum(CASE WHEN (cs_ship_date_sk - cs_sold_date_sk <= 30) THEN 1 ELSE 0 END) AS "30 days", sum(CASE WHEN (cs_ship_date_sk - cs_sold_date_sk > 30) AND (cs_ship_date_sk - cs_sold_date_sk <= 60) THEN 1 ELSE 0 END) AS "31-60 days", sum(CASE WHEN (cs_ship_date_sk - cs_sold_date_sk > 60) AND (cs_ship_date_sk - cs_sold_date_sk <= 90) THEN 1 ELSE 0 END) AS "61-90 days", sum(CASE WHEN (cs_ship_date_sk - cs_sold_date_sk > 90) AND (cs_ship_date_sk - cs_sold_date_sk <= 120) THEN 1 ELSE 0 END) AS "91-120 days", sum(CASE WHEN (cs_ship_date_sk - cs_sold_date_sk > 120) THEN 1 ELSE 0 END) AS ">120 days" FROM catalog_sales , (SELECT SUBSTR(w_warehouse_name,1,20) w_substr, * FROM warehouse) AS sq1 , ship_mode , call_center WHERE cs_ship_date_sk BETWEEN 2451545 AND 2451910 AND cs_warehouse_sk = w_warehouse_sk AND cs_ship_mode_sk = sm_ship_mode_sk AND cs_call_center_sk = cc_call_center_sk GROUP BY w_substr , sm_type , cc_name ORDER BY w_substr , sm_type , cc_name LIMIT 100;"""
}
