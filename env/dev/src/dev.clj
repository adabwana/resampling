(ns dev
  (:require [scicloj.clay.v2.api :as clay]))

(defn build []
  (clay/make!
    {:format           [:quarto :html]
     :book             {:title "Resampling, Selection, & Regularization"}
     :base-source-path "src"
     :base-target-path "docs"
     :subdirs-to-sync  ["notebooks" "data"]
     :source-path      ["index.clj"                         ;index.md
                        "assignment/eda.clj"
                        "assignment/scicloj.clj"]}))

(comment
  ;with index.md clay wont find in src and complains about docs/_book
  (build))