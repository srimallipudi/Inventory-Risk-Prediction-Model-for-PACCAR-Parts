library(vioplot)
library(ggplot2)
library(tidyverse)
library(patchwork)
library(lubridate)
library(gridExtra)


rm(list=ls(all=TRUE)) #clears memory
options(digits=12) #keeps 12 digits in memory, otherwise summary statistics may be off due to rounding

#Set working directory by clicking on Session --> set working directory --> to source file location

data <- read.csv("test_data_0727_1031.csv", header=TRUE)  # imports and renames dataset

# Subset the data for stockout and non-stockout periods
stockout_data <- data[data$rhit_label == 1, ]
non_stockout_data <- data[data$rhit_label != 1, ]

# Create a box plot comparing lead_time for stockout and non-stockout periods
boxplot(list(stockout_data$lead_time, non_stockout_data$lead_time),
        names = c("Stockout", "Non-Stockout"),
        xlab = "Period",
        ylab = "Lead Time",
        main = "Lead Time Distribution: Stockout vs Non-Stockout",
        col = c("red", "blue"),
        notch = TRUE,       # Add notches to the box plots for a visual comparison of medians
        notchwidth = 0.5,   # Adjust the width of the notches
        outline = FALSE)  # Remove outliers from the plot for better focus on the distributions

# Add legend
legend("topright", legend = c("Stockout", "Non-Stockout"), fill = c("red", "blue"))

new <- stockout_data %>%
  filter(rhit_label == 1) %>%
  mutate(OHlessSS = case_when(
    doh_less_ss == 1 ~ 'On hand < Safety Stock ',
    TRUE ~ 'On hand > Safety Stock'
  )) %>%
  summarise(Total = n(),
            `On hand < Safety Stock` = sum(doh_less_ss)/n(),
            `On hand > Safety Stock` = sum(1 - doh_less_ss)/n()) %>%

ggplot(new, aes(x = OHlessSS, fill = OHlessSS)) + geom_bar()