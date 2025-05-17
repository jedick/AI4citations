# Compile SciFact test data to evaluate retrieval in AI4citations
# 20250513 jmd

# Load package to read jsonl files
# https://stackoverflow.com/questions/35016713/how-do-i-import-a-jsonl-file-in-r-and-how-do-i-transform-it-in-csv
library(jsonlite)

# Read the claims file
claims_file <- "../data/scifact/claims_test.jsonl"
claims_df <- stream_in(file(claims_file))

# Get the label for each claim
label <- sapply(1:nrow(claims_df), function(iclaim) {
  this_evidence <- claims_df$evidence[iclaim, ]
  # Take out sentences and keep labels
  evidence_pieces <- unlist(this_evidence)
  ilabel <- grep("label", names(evidence_pieces))
  label <- unique(evidence_pieces[ilabel])
  if(is.null(label)) label <- "NEI"
  label
})

# Get the claims and cited doc IDs
claim <- claims_df$claim
# "Corpus ID" is the name used in Semantic Scholar
corpus_id <- unlist(claims_df$cited_doc_ids)

# Now read the corpus file
corpus_file <- "../data/scifact/corpus.jsonl"
corpus_df <- stream_in(file(corpus_file))

# Match cited doc IDs to corpus
icorpus <- match(corpus_id, corpus_df$doc_id)
# Get the titles and abstracts
title <- corpus_df$title[icorpus]
abstract <- corpus_df$abstract[icorpus]
# Join sentences in each abstract
abstract <- sapply(abstract, paste, collapse = " ")
# Remove newlines from abstracts
abstract <- sapply(abstract, gsub, pattern = "\n", replacement = " ")

# Create output
output_df <- data.frame(claim, label, corpus_id, title, abstract)
write.csv(output_df, "scifact_test_data.csv", row.names = FALSE)

## Save table of unique source docs
#idup <- duplicated(output_df$corpus_id)
#unique_docs <- output_df[!idup, 3:4]
#write.csv(unique_docs, "scifact_test_data_sources.csv", row.names = FALSE)
