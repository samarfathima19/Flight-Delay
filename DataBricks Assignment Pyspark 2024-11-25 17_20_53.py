# Databricks notebook source
# MAGIC %md
# MAGIC ## Dataset Description

# COMMAND ----------

# MAGIC %md
# MAGIC The **2015 Flight Delays and Cancellations** dataset contains detailed records of flight information for the year 2015. It includes data related to flight delays, cancellations, and their causes, along with information about the airlines and airports involved. 
# MAGIC
# MAGIC **Dataset Link :** https://www.kaggle.com/datasets/usdot/flight-delays
# MAGIC
# MAGIC This dataset consists of three CSV files:
# MAGIC
# MAGIC **1. airlines.csv**
# MAGIC
# MAGIC **IATA_CODE:** Airline Identifier
# MAGIC
# MAGIC **AIRLINE:** Name of the airline
# MAGIC
# MAGIC **2. airports.csv**
# MAGIC
# MAGIC **IATA_CODE:** Airport Identifier
# MAGIC
# MAGIC **AIRPORT:** Name of the airport
# MAGIC
# MAGIC **CITY:** City of the airport
# MAGIC
# MAGIC **STATE:** State of the airport
# MAGIC
# MAGIC **COUNTRY:** Country of the airport
# MAGIC
# MAGIC **LATITUDE:** Latitude of the airport
# MAGIC
# MAGIC **LONGITUDE:** Longitude of the airport
# MAGIC
# MAGIC **3. flights.csv**
# MAGIC
# MAGIC **YEAR:** Year of the flight trip
# MAGIC
# MAGIC **MONTH:** Month of the flight trip
# MAGIC
# MAGIC **DAY:** Day of the month
# MAGIC
# MAGIC **DAY_OF_WEEK:** Day of the week
# MAGIC
# MAGIC **AIRLINE:** Airline Identifier
# MAGIC
# MAGIC **FLIGHT_NUMBER:** Flight Identifier
# MAGIC
# MAGIC **TAIL_NUMBER:** Aircraft Identifier
# MAGIC
# MAGIC **ORIGIN_AIRPORT:** Starting airport code
# MAGIC
# MAGIC **DESTINATION_AIRPORT:** Destination airport code
# MAGIC
# MAGIC **SCHEDULED_DEPARTURE:** Scheduled departure time
# MAGIC
# MAGIC **DEPARTURE_TIME:** Actual departure time
# MAGIC
# MAGIC **DEPARTURE_DELAY:** Delay on departure
# MAGIC
# MAGIC **TAXI_OUT:** Time between origin airport gate and wheels off
# MAGIC
# MAGIC **WHEELS_OFF:** Time when the aircraft’s wheels leave the ground
# MAGIC
# MAGIC **SCHEDULED_TIME:** Planned time for the flight trip
# MAGIC
# MAGIC **ELAPSED_TIME:** Total flight time
# MAGIC
# MAGIC **AIR_TIME:** Time between wheels off and wheels on
# MAGIC
# MAGIC **DISTANCE:** Distance between origin and destination airports
# MAGIC
# MAGIC **WHEELS_ON:** Time when the aircraft’s wheels touch the ground
# MAGIC
# MAGIC **TAXI_IN:** Time from wheels on to gate arrival
# MAGIC
# MAGIC **SCHEDULED_ARRIVAL:** Scheduled arrival time
# MAGIC
# MAGIC **ARRIVAL_TIME:** Actual arrival time
# MAGIC
# MAGIC **ARRIVAL_DELAY:** Delay on arrival
# MAGIC
# MAGIC **DIVERTED:** Whether the flight was diverted (1 = diverted)
# MAGIC
# MAGIC **CANCELLED:** Whether the flight was cancelled (1 = cancelled)
# MAGIC
# MAGIC **CANCELLATION_REASON:** Reason for cancellation (A = Airline, B = Weather, C = National Air System, D = Security)
# MAGIC
# MAGIC **AIR_SYSTEM_DELAY:** Delay caused by air system
# MAGIC
# MAGIC **SECURITY_DELAY:** Delay caused by security
# MAGIC
# MAGIC **AIRLINE_DELAY:** Delay caused by the airline
# MAGIC
# MAGIC **LATE_AIRCRAFT_DELAY:** Delay caused by aircraft
# MAGIC
# MAGIC **WEATHER_DELAY:** Delay caused by weather

# COMMAND ----------

# MAGIC %md
# MAGIC Data Loading and Initial Exploration

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("FlightDelay").getOrCreate()

# COMMAND ----------

# Loading the datasets
flights_df = spark.read.csv("dbfs:/FileStore/flights.csv", header=True, inferSchema=True)
airlines_df = spark.read.csv("dbfs:/FileStore/airlines.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("dbfs:/FileStore/airports.csv", header=True, inferSchema=True)

# COMMAND ----------

# Rename the AIRLINE column in the airlines_df to avoid conflicts
airlines_df = airlines_df.withColumnRenamed("AIRLINE", "AIRLINE_NAME")

# Join flights with airlines
flights_airlines_df = flights_df.join(airlines_df, flights_df.AIRLINE == airlines_df.IATA_CODE, "left")

# Rename IATA_CODE in airports_df to avoid conflicts
airports_origin = airports_df.withColumnRenamed("IATA_CODE", "ORIGIN_AIRPORT_CODE")
airports_destination = airports_df.withColumnRenamed("IATA_CODE", "DESTINATION_AIRPORT_CODE")

# Join with origin airport details
flights_with_origin = flights_airlines_df.join(
    airports_origin, 
    flights_df.ORIGIN_AIRPORT == airports_origin.ORIGIN_AIRPORT_CODE, 
    "left"
)

# Join with destination airport details
flights_with_destination = flights_with_origin.join(
    airports_destination, 
    flights_df.DESTINATION_AIRPORT == airports_destination.DESTINATION_AIRPORT_CODE, 
    "left"
)

# The final dataframe with all the columns joined
df = flights_with_destination

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# Drop redundant columns
df = df.drop("IATA_CODE", "AIRLINE", "ORIGIN_AIRPORT_CODE", "DESTINATION_AIRPORT_CODE", 
             "AIRPORT", "CITY", "STATE", "COUNTRY", "LATITUDE", "LONGITUDE")

# COMMAND ----------

# MAGIC %md
# MAGIC **Q1.  What are the data types of each column?**

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC **Q2. How many rows and columns does the dataset contain?**

# COMMAND ----------

# Count the number of rows
row_count = df.count()

# Count the number of columns
column_count = len(df.columns)

print(f"Number of rows: {row_count}")
print(f"Number of columns: {column_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Cleaning and Transformation 

# COMMAND ----------

# MAGIC %md
# MAGIC **Q3. Which columns have missing values, and how can you handle them?**

# COMMAND ----------

from pyspark.sql.functions import col, count, when

# Counting nulls in each column
missing_values = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])

# Collect the result as a list
missing_values_list = missing_values.collect()[0].asDict()

# Print the missing values with the column name
print("Missing values per column:")
for column, missing_count in missing_values_list.items():
    print(f"{column}: {missing_count}")

# COMMAND ----------

columns_to_drop = [
    'CANCELLATION_REASON', 'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 
    'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'
]
df = df.drop(*columns_to_drop)

# COMMAND ----------

# MAGIC %md
# MAGIC **Dropping columns with excessive missing values since these columns will not be useful for analysis**

# COMMAND ----------

df = df.dropna(subset=['TAIL_NUMBER'])

# COMMAND ----------

# MAGIC %md
# MAGIC **Missing values in TAIL_NUMBER are dropped because this is a crucial identifier for aircraft, and rows without it might not be reliable**

# COMMAND ----------

df = df.fillna({
    'SCHEDULED_TIME': 0,
    'ELAPSED_TIME': 0,
    'WHEELS_OFF': 0,
    'WHEELS_ON': 0,
    'DEPARTURE_TIME': 0,
    'DEPARTURE_DELAY': 0,  
    'ARRIVAL_DELAY': 0,    
    'AIR_TIME': 0,         
    'TAXI_OUT': 0,        
    'TAXI_IN': 0,          
    'ARRIVAL_TIME': 0,     
})

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **There are a lot of missing values in these columns. Since missing values in this case could indicate no recorded data (perhaps as a result of flight cancellations or schedule conflicts), we can fill them in with 0.**
# MAGIC

# COMMAND ----------

# Counting nulls in each column
updated_missing_values = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])

# Collect the result as a list
updated_missing_values_list = updated_missing_values.collect()[0].asDict()

# Print the missing values in a schema-like format
print("Missing values per column:")
for column, updated_missing_count in updated_missing_values_list.items():
    print(f"{column}: {updated_missing_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Q4. What new features did you create, and why?**

# COMMAND ----------

import pyspark.sql.functions as F
def get_time_of_day(departure_time):
    hour = (departure_time // 100) % 24
    if 5 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 17:
        return 'Afternoon'
    elif 17 <= hour < 21:
        return 'Evening'
    else:
        return 'Night'

df = df.withColumn('TIME_OF_DAY', F.udf(get_time_of_day)(F.col('SCHEDULED_DEPARTURE')))

# COMMAND ----------

# MAGIC %md
# MAGIC **Categorizing the SCHEDULED_DEPARTURE and DEPARTURE_TIME into time of day (Morning, Afternoon, Evening, Night). This may help capture time-of-day-specific patterns**

# COMMAND ----------

df = df.withColumn('TOTAL_DELAY', F.col('DEPARTURE_DELAY') + F.col('ARRIVAL_DELAY'))

# COMMAND ----------

# MAGIC %md
# MAGIC **This feature will give a comprehensive understanding of the overall delay, incorporating both departure and arrival delays. It's crucial for predicting delays and understanding how they affect flight performance.**

# COMMAND ----------

df = df.withColumn('EFFICIENCY_RATIO', F.col('AIR_TIME') / F.col('DISTANCE'))

# COMMAND ----------

# MAGIC %md
# MAGIC **This feature measures how efficiently a flight was executed. It can reveal insights into potential delays caused by inefficient flights, such as slower-than-usual travel or unusual detours. It's particularly useful in understanding flight patterns and anomalies.**

# COMMAND ----------

display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis (EDA) using Spark SQL

# COMMAND ----------

# Register DataFrame as a temporary SQL table
df.createOrReplaceTempView("flights_table")

# COMMAND ----------

# MAGIC %md
# MAGIC **Q5. Calculate the average delay time for each airline.**

# COMMAND ----------

average_delay_query = """
SELECT AIRLINE_NAME, ROUND(AVG(DEPARTURE_DELAY), 3) AS AVG_DEPARTURE_DELAY
FROM flights_table
GROUP BY AIRLINE_NAME
ORDER BY AVG_DEPARTURE_DELAY DESC
"""

average_delay_df = spark.sql(average_delay_query)
average_delay_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Q6. Identify the top 5 airports with the most delayed departures.**

# COMMAND ----------

top_airports_query = """
SELECT ORIGIN_AIRPORT, SUM(DEPARTURE_DELAY) AS TOTAL_DEPARTURE_DELAY
FROM flights_table
WHERE DEPARTURE_DELAY IS NOT NULL
GROUP BY ORIGIN_AIRPORT
ORDER BY TOTAL_DEPARTURE_DELAY DESC
LIMIT 5
"""

top_airports_df = spark.sql(top_airports_query)
top_airports_df.show() 

# COMMAND ----------

# MAGIC %md
# MAGIC **Q7. Determine the most common reason for flight cancellations.**

# COMMAND ----------

cancelled_delays_query = """
SELECT 
    CASE 
        WHEN DEPARTURE_DELAY > 120 THEN 'High Departure Delay'
        WHEN ARRIVAL_DELAY > 120 THEN 'High Arrival Delay'
        WHEN DIVERTED = 1 THEN 'Flight Diverted'
        ELSE 'Other Reasons'
    END AS CANCELLATION_REASON,
    COUNT(*) AS CANCELLED_FLIGHTS
FROM flights_table
WHERE CANCELLED = 1
GROUP BY CANCELLATION_REASON
ORDER BY CANCELLED_FLIGHTS DESC
"""

cancelled_delays_df = spark.sql(cancelled_delays_query)
cancelled_delays_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Note** I experimented with several features for my analysis because I had already removed the CANCELLATION_REASON column from my dataset due to the large number of missing values.
# MAGIC
# MAGIC I examined DEPARTURE_DELAY, ARRIVAL_DELAY, AIR_TIME, and DISTANCE for flights that were canceled because they may be the result of major delays.
# MAGIC
# MAGIC Given that many planes that are diverted are subsequently canceled, DIVERTED may offer further information.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC **Q8. Which airlines had the highest average delays?**

# COMMAND ----------

import matplotlib.pyplot as plt

# Convert the PySpark DataFrame to Pandas DataFrame
avg_delay_pd = average_delay_df.toPandas()

# Plotting the data
plt.figure(figsize=(10, 6))
plt.barh(avg_delay_pd['AIRLINE_NAME'], avg_delay_pd['AVG_DEPARTURE_DELAY'], color='brown')
plt.xlabel('Average Departure Delay (Minutes)')
plt.ylabel('Airline Name')
plt.title('Top 10 Airlines with the Highest Average Departure Delays')
plt.gca().invert_yaxis()  # To display the highest delay on top
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Insights :-**
# MAGIC
# MAGIC
# MAGIC With departure delays of more than ten minutes on average, Spirit Airlines has the longest average, followed by United Air Lines Inc. and Frontier Airlines Inc. This implies that prompt departures may be a persistent problem for certain carriers.
# MAGIC
# MAGIC Despite being lower than the top three, Southwest Airlines Co. and jetBlue Airways also exhibit very high average delays.
# MAGIC
# MAGIC With average delays of about three to four minutes, Alaska carriers Inc. and Hawaiian Airlines Inc. exhibit comparatively smaller delays than the other carriers in the top 10.
# MAGIC
# MAGIC The disparities among the airlines raise the possibility that some may have more operational or logistical difficulties than others.
# MAGIC
# MAGIC The top 5 airlines show a significant difference in their average delays compared to the lower half of the top 10, indicating that a few airlines are significantly more prone to delays.

# COMMAND ----------

# MAGIC %md
# MAGIC **Q9. What patterns did you observe in delays by day of the week?**

# COMMAND ----------

# SQL Query to calculate average delay by day of the week
query = """
SELECT DAY_OF_WEEK, AVG(DEPARTURE_DELAY) AS AVG_DEPARTURE_DELAY
FROM flights_table
WHERE CANCELLED = 0 AND DEPARTURE_DELAY IS NOT NULL
GROUP BY DAY_OF_WEEK
ORDER BY DAY_OF_WEEK
"""

avg_delay_by_day_df = spark.sql(query)

# COMMAND ----------

avg_delay_by_day_pd = avg_delay_by_day_df.toPandas()

plt.figure(figsize=(10, 6))
plt.bar(avg_delay_by_day_pd['DAY_OF_WEEK'], avg_delay_by_day_pd['AVG_DEPARTURE_DELAY'], color='brown')
plt.xlabel('Day of the Week')
plt.ylabel('Average Departure Delay (Minutes)')
plt.title('Average Departure Delay by Day of the Week')
plt.xticks(ticks=range(1, 8), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Observations:-**
# MAGIC
# MAGIC The average departure delay on Monday is the highest, with a significant peak, indicating that Monday flights typically have lengthier delays than other days.
# MAGIC
# MAGIC Tuesday through Friday have comparatively similar delay periods, all of which exhibit modest delays, suggesting a generally stable pattern throughout these days.
# MAGIC
# MAGIC When compared to weekdays, Saturday and Sunday both exhibit somewhat reduced average delays, with Sunday's delay length about matching Friday's.
# MAGIC
# MAGIC The operational strain at the end of the workweek may be reflected in the similar average delay between Thursday and Friday, which could be caused by busier travel periods.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary

# COMMAND ----------

# MAGIC %md
# MAGIC * The visuals and SQL searches consistently show that airlines with high average delays (Frontier, United, and Spirit) typically have greater delays.
# MAGIC
# MAGIC * Major hub airports like ORD, ATL, and DFW rank first in terms of delayed departures, suggesting that larger airports with higher traffic volumes may experience more operational difficulties and delays.
# MAGIC
# MAGIC * While weekends often have somewhat fewer delays, Monday has the most, possibly as a result of increased demand following the weekend.
# MAGIC
# MAGIC * The majority of flight cancellations are caused by factors other than delays; substantial delays only make up a minor percentage of cancellations, with Other Reasons accounting for the largest share.
# MAGIC

# COMMAND ----------


