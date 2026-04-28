# Overwintering
Wildland fire fighters have reported annecdotal evidence of multiyear overwintering fires in Northeastern BC boreal plains based on in the field observations. To support this field-based observation evidence, we must provide some quantitative evidence. Satelitte based observation can help to provide support. The Fire Information for Resource Management System (FIRMS) distributes near-real-time (NRT) active fire observation in Canada from the Visible Infared Imaging Radiometer Suite (VIIRS) aboard S-NPP, NOAA 20 and NOAA 21. Using these fire hotspot and the fire perimeters distributed by the BC government for years 2023-2025, we are able to monitor the likelihood that a hotspot in the year following the initial fire (in this case 2023) occured within the boundary(1000m buffer) of the 2023 fire perimeter. To ensure that this is an overwintering fire, we search for hotspots in the following spring that occur after snowmelt. Using this logic, we following the same pattern but for 2024-2025 to identify fires that started burning in 2023, went dorment in winter, came back and started burning following snowmelt in spring 2024, went dorment again in winter (2024-2025) and started burning following snowmelt in spgin 2025. 


## Note: 
we did not use MODIS for the following reasons: ....

## Hotspot considerations:
- daynight flags: we should consider restricting analysis to daynight == 'N', daytime detections are contaminated by solar reflection and the nightime retrievals use the M13 band at 4um without solar interference, giving cleaner thermal signal
-confidence flags: l (low), n (nominal), h (high) for detection confidence, may be worth using only n and h, but good to know the distribution
- could be better to weight observations by fire radiative power (FRP)
- consider filtering by bright_ti4
- might be worth sampling to near-nadir pixels (sample column, 400-3200) because detection near swath edge have larger, distored pixels and higher false positive rates. 

## Prep: 
- make sure Fire Zone is a column per fire perimeter
- find fire perimeters that overlap in subsequent years
- Night-only hotspot detections (daynight == 'N') 
- drop low (l) hotspot confidence
- scan and track column filtering to replicate near-nadir scan <=0.5 and track <=0.5
- Snowmelt onset day calcualted for each fire in 2023-205. 



## Snowmelt onset day (Snow disappearance date)
Average snowmelt onset day for each fire zone calculated from a combined MODIS Acqua/Terra daily fractional snow cover product at 500m resolution. Processing followed the the snow cloud metrics algorithm for SDD. Algorithm was adjusted for the Jan 1 to July 1. 
	Crumley, R. L., Palomaki, R. T., Nolin, A. W., Sproles, E. A., & Mar, E. J. (2020). SnowCloudMetrics: Snow Information for Everyone. Remote Sensing, 12(20), 3341. https://doi.org/10.3390/rs12203341


**Snowmelt onset DOY average per fire zone**
| Fire Zone | Year | Snowmelt onset DOY  |
| --------- | ---- | ----------------- | 
| Fort St. John | 2024 | 135.71 |
| Fort Nelson | 2024 |165.14 |
| Fort St. John | 2025 | 140.63 |
| Fort Nelson | 2025 | 167.72 |



## Algorithm for overwintering fires steps:
1. create buffer around fire in 2023 and 2024 at 1000m based on scholten et al. (consider sensativity analysis with 500m and 2000m)
2. Identify the hotspots that are within the 2023 and 2024 perimeters in the fall before winter.
3. Identify the hotspots from 2024 and 2025 that are within the 2023 and 2024 buffered perimeters in the spring following snowmelt onset day. 
4. Classify fires based on:
  - 2023-2024 overwinter (2024 spring hotspot in buffer)
  - 2023-2024 non-overwinter (2024 spring hotspot not in buffer)
  - 2024-2025 overwinter (2025 spring hotspot in buffer)
  - 2024-2025 non-overwinter (2025 spring hotspot not in buffer)

**PERSISTENCE CHECK** for a fire to be classified as overwintering, fall hotspot cluster has a minimum number of detections (e.g., >2-3 unique days) within the perimter before winter. Single-detection fall hotspots have higher false positive rates and may not represent genuine residual burning.

5. Find fires that overwintered in 2023-2024 and 2024-2025 (multi-year overwintering)
  - 2025 hotspots in 2024 buffer where the 2024 perim has a hotspot in 2023 buffer
6. For each case create an individual dataframe:
  - column hotspot 2023, 2024 and 2025
  - column for hotspot location (takes either buffer, perimeter) based on its location 
  - hotspot lat and lon
  - date of hotspot detection
  - satelitte of hotspot
  - confidence level
  - day/nighttime
7. Find latest 2023 (2024) hotspot and earliest 2024 (2025) hotspot and calculate
  - euclidean distance between two
  - time difference between two
  - distance of spring hotspot to previous year's perimeter
  - time difference between spring hotspot and snowmelt onset day
  - formally define the dormancy window - e.g., no detectable hotspot within the perimter between Nov 1 and snowmelt onset DOY.
8. Summary statistics for each fire zone and year:
  - number of overwintering fires
  - number of non-overwintering fires
  - number of multi-year overwintering fires
  - average distance between latest and earliest hotspots
  - average time difference between latest and earliest hotspots
  - average distance of spring hotspot to previous year's perimeter
  - average time difference between spring hotspot and snowmelt onset day







