#### Run SpaGene ####
# Read the file storing the scanpy data, and use SpaGene to identify the spagene
#, and then store it.


# 1. Import ----------------------
args <- commandArgs(trailingOnly = TRUE) # nolint

library(rhdf5)
library(Matrix)
library(SpaGene)

# num:default=200 # nolint

dirname <- args[1]
num <- ifelse(length(args) == 1, 200, as.integer(args[2]))

# 2. Read ------------------------
count_path <- paste(dirname, "count.h5", sep = "/")
mat <- h5read(count_path, "count")
count <- mat$block0_values
rownames(count) <- mat$axis0
colnames(count) <- mat$axis1
count <- Matrix(count, sparse = TRUE)

location_path <- paste(dirname, "spatial.tsv", sep = "/")
location <- read.table(location_path, sep = "\t", header = TRUE, row.names = 1)

# 3. SpaGene ---------------------
result <- SpaGene(count, location)
result <- result$spagene_res[order(result$spagene_res$adjp), ]
max_num <- sum(result["adjp"] < 0.01)
num <- min(max_num, num)
gene <- rownames(result)[1:num]

# 4. Write -----------------------
write_path <- paste(dirname, "spagene.txt", sep = "/")
write(gene, file = write_path)
