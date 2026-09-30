# Export the source datasets of the benchmark from their R packages to benchmarks/data/ (CSV).
# The files are not redistributed with the repository; run this once, then python benchmarks/prepare_data.py.
# Packages: nlme (Remifentanil) and nlmixr2data 2.0.10 (pheno_sd, Oral_1CPTMM).
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("^--file=", "", args[grep("^--file=", args)])))
out <- file.path(here, "data")
dir.create(out, showWarnings = FALSE)
suppressMessages({
  library(nlme)
  library(nlmixr2data)
})
remi <- as.data.frame(nlme::Remifentanil)
remi$Sex <- as.integer(remi$Sex == "Male")                       # 1 male, 0 female
remi <- remi[, c("ID", "Time", "conc", "Rate", "Amt", "Age", "Sex", "Ht", "Wt", "BSA", "LBM")]
write.csv(remi, file.path(out, "remifentanil_nlme.csv"), row.names = FALSE, na = ".")
write.csv(nlmixr2data::pheno_sd, file.path(out, "pheno_sd.csv"), row.names = FALSE)
write.csv(nlmixr2data::Oral_1CPTMM, file.path(out, "Oral_1CPTMM.csv"), row.names = FALSE)
cat("nlme", as.character(packageVersion("nlme")), "nlmixr2data", as.character(packageVersion("nlmixr2data")), "\n")
