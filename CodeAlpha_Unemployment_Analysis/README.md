\# CodeAlpha Unemployment Analysis



\## Project Overview



This project analyzes unemployment rate data in India using Python and

explores unemployment trends, regional differences, rural versus urban

unemployment, and the impact of the COVID-19 period.



\## Objective



The main objectives of this project are:



\- Clean and prepare the unemployment dataset.

\- Explore unemployment rate statistics.

\- Analyze unemployment trends over time.

\- Investigate the impact of COVID-19 on unemployment.

\- Compare unemployment rates across regions.

\- Compare rural and urban unemployment.

\- Analyze monthly unemployment patterns.

\- Visualize important findings using graphs.



\## Technologies Used



\- Python

\- Pandas

\- Matplotlib

\- CSV Dataset



\## Dataset



The dataset contains unemployment information for different regions

of India, including:



\- Region

\- Date

\- Frequency

\- Estimated Unemployment Rate (%)

\- Estimated Employed

\- Estimated Labour Participation Rate (%)

\- Area



\## Data Cleaning



The dataset originally contained 768 rows.



After removing completely empty rows, 740 valid records remained.



The following preprocessing steps were performed:



1\. Removed completely empty rows.

2\. Removed unnecessary whitespace from column names.

3\. Cleaned text values.

4\. Converted the Date column into datetime format.

5\. Checked for missing values.

6\. Saved the cleaned dataset.



\## Analysis Performed



\### 1. Overall Unemployment Trend



The monthly average unemployment rate was calculated to understand

changes over time.



\### 2. COVID-19 Impact



The unemployment rate was compared between the pre-COVID period and

the COVID period.



\- Pre-COVID average: 9.51%

\- COVID-period average: 17.77%

\- Increase: 8.26 percentage points



\### 3. Regional Analysis



Average unemployment rates were calculated for 28 regions.



The dataset shows substantial differences in average unemployment

rates across regions.



\### 4. Rural vs Urban Analysis



The average unemployment rates were:



\- Rural: 10.32%

\- Urban: 13.17%



\### 5. Monthly Pattern



The monthly average unemployment rates show a sharp increase during

April and May 2020.



The highest monthly average in this dataset was:



\- April: 23.64%



\## Visualizations



The project generates the following visualizations:



1\. Overall unemployment trend

2\. COVID-19 unemployment comparison

3\. Regional unemployment comparison

4\. Rural vs Urban unemployment comparison

5\. Monthly unemployment pattern



\## Key Findings



\- Unemployment increased substantially during the COVID-19 period.

\- The average unemployment rate increased from 9.51% before COVID to

&#x20; 17.77% during the COVID period.

\- Urban unemployment averaged 13.17%, compared with 10.32% in rural

&#x20; areas in this dataset.

\- April 2020 recorded the highest monthly average unemployment rate

&#x20; at 23.64%.

\- Unemployment rates varied considerably across the 28 regions.



\## Project Structure



```text

CodeAlpha\_Unemployment\_Analysis/

│

├── data/

│   ├── Unemployment in India.csv

│   └── Unemployment in India\_cleaned.csv

│

├── output/

│   ├── overall\_unemployment\_trend.png

│   ├── covid\_unemployment\_comparison.png

│   ├── regional\_unemployment.png

│   ├── rural\_vs\_urban\_unemployment.png

│   └── monthly\_unemployment\_pattern.png

│

├── unemployment\_analysis.py

└── README.md

