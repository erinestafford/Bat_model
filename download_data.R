setwd("~/Documents/Github/Bat_model")
library(raster)
thai_data =raster("GLCFCS30_E100N15.tif")
legend <- read.csv(text="Value,Label,Red,Green,Blue
                    10,	Rainfed cropland,	255,255,100
                    11,	Herbaceous cover,	255,255,100
                    12,	Tree or shrub cover (Orchard),	255,255,0
                    20,	Irrigated cropland,	170,240,240
                    50,	Evergreen broadleaved forest,	0,100,0
                    60,	Deciduous broadleaved forest,	0,160,0
                    61,	Open deciduous broadleaved forest (0.15<fc<0.4),	0,160,0
                    62,	Closed deciduous broadleaved forest (fc>0.4),	170,200,0
                    70,	Evergreen needle-leaved forest,	0,60,0
                    71,	Open evergreen needle-leaved forest (0.15< fc <0.4),	0,60,0
                    72,	Closed evergreen needle-leaved forest (fc >0.4),	0,80,0
                    80,	Deciduous needle-leaved forest,	40,80,0
                    81,	Open deciduous needle-leaved forest (0.15< fc <0.4),	40,80,0
                    82,	Closed deciduous needle-leaved forest (fc >0.4),	40,100,0
                    90,	Mixed leaf forest (broadleaved and needle-leaved),	20,130,0
                    120,	Shrubland,	150,100,0
                    121, Evergreen shrubland,	150,75,0
                    122,	Deciduous shrubland,	150,100,0
                    130,	Grassland,	255,180,50
                    140,	Lichens and mosses,	255,220,210
                    150,	Sparse vegetation (fc<0.15),	255,235,175
                    152,	Sparse shrubland (fc<0.15),	255,210,120
                    153,	Sparse herbaceous (fc<0.15),	255,235,175
                    180,	Wetlands,	0,220,130
                    190,	Impervious,	195,20,0
                    200,	Bare areas,	255,245,215
                    201,	Consolidated bare areas,	220,220,220
                    202,	Unconsolidated bare areas,	255,245,215
                    210,	Water body,	0,70,200
                    220,	Permanent ice and snow,	255,255,255
                    250,	Filled value,	255,255,255")
legend$col <-  rgb(legend$Red, legend$Green, legend$Blue, maxColorValue=255)
tb <- legend[, c('Value', 'Label')]
colnames(tb)[1] = "ID"
tb$Label <- substr(tb$Label, 1,10)
levels(thai_data) <- tb

library(rasterVis)
w10 <- thai_data[6700:8700,8000:10000, drop=FALSE]
levelplot(w10, col.regions=legend$col)

writeRaster(w10,'test.tif',options=c('TFW=YES'),overwrite=TRUE)

#library(terra)
#m <- as.matrix(w10, wide = TRUE)
#dim(m)
#write.csv(m, "Thai_LU_data.csv")


#thai_data2 =raster("test.tif")
#levelplot(thai_data2, col.regions=legend$col)
