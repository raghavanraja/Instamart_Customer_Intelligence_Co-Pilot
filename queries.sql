-- 1. Customer and order volume
SELECT COUNT(DISTINCT customer_id) AS customers,
       COUNT(*) AS orders
FROM orders;

-- 2. Revenue and order count by city
SELECT c.city, COUNT(*) AS order_count,
       ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o JOIN customers c USING(customer_id)
WHERE o.status = 'completed'
GROUP BY c.city ORDER BY revenue_inr DESC;

-- 3. Revenue by product category
SELECT p.category, COUNT(*) AS line_count,
       ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o JOIN products p USING(product_id)
WHERE o.status = 'completed'
GROUP BY p.category ORDER BY revenue_inr DESC;

-- 4. Monthly completed-order trend
SELECT substr(order_ts, 1, 7) AS month,
       COUNT(*) AS completed_orders,
       ROUND(SUM(order_amount), 2) AS revenue_inr
FROM orders WHERE status='completed'
GROUP BY month ORDER BY month;

-- 5. Late completed orders
SELECT COUNT(*) AS late_orders,
       ROUND(AVG(delivery_minutes), 2) AS average_delivery_minutes
FROM orders
WHERE status='completed' AND delivery_minutes > 45;

-- 6. Top 10 customers by completed spend
SELECT customer_id, COUNT(*) AS completed_orders,
       ROUND(SUM(order_amount), 2) AS spend_inr
FROM orders WHERE status='completed'
GROUP BY customer_id ORDER BY spend_inr DESC LIMIT 10;

-- 7. Customer order status summary
SELECT status, COUNT(*) AS order_count
FROM orders GROUP BY status ORDER BY status;
