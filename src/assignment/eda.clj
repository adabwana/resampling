(ns assignment.eda
  (:require
    [clojure.math.combinatorics :as combo]
    [fastmath.stats :as stats]
    [scicloj.kindly.v4.kind :as kind]
    [scicloj.ml.dataset :as ds]))

;; # Exploratory Data Analysis
;; Load data
(defonce boston
         (ds/dataset "data/boston.csv"
                     {:key-fn (fn [colname]
                                (-> colname
                                    (clojure.string/replace #"\.|\s" "-")
                                    clojure.string/lower-case
                                    keyword))}))

(ds/info boston)

(def response :medv)
(def regressors
  (ds/column-names boston (complement #{response})))

;; ## Raw
;; ### Histograms
^kind/vega
(let [data (ds/rows boston :as-maps)
      column-names (ds/column-names boston)]
  {:data   {:values data}
   :repeat {:column column-names}
   :spec   {:mark     "bar"
            :encoding {:x {:field {:repeat "column"} :type "quantitative"}
                       :y {:aggregate "count"}}}})

;; ### Box plots
^kind/vega
(let [data (ds/rows boston :as-maps)
      column-names (ds/column-names boston)]
  {:data   {:values data}
   :repeat {:column column-names}
   :spec   {:width    60 :mark "boxplot"
            :encoding {:y {:field {:repeat "column"} :type "quantitative" :scale {:zero false}}}}})

;; #### Outliers
(let [columns (ds/column-names boston)]
  (->> (for [column columns]
         (vector column (count (stats/outliers (get boston column)))))
       (sort-by first)))

;; ### Pairs plot
^kind/vega
(let [data (ds/rows boston :as-maps)
      column-names (ds/column-names boston)]
  {:data   {:values data}
   :repeat {:column column-names
            :row    column-names}
   :spec   {:height   100 :width 100
            :mark     "circle"
            :encoding {:x {:field {:repeat "column"} :type "quantitative" :scale {:zero false}}
                       :y {:field {:repeat "row"} :type "quantitative" :scale {:zero false}}}}})

(let [combos (combo/combinations regressors 2)]
  (for [[x y] combos]
    (assoc {} [x y] (stats/correlation (get boston x) (get boston y)))))

(for [[x y] (mapv (fn [r] [response r]) regressors)]
  (assoc {} [x y] (stats/correlation (get boston x) (get boston y))))

;; ## Standardized
(defn standardize-column [dataset]
  (reduce (fn [acc key]
            (assoc acc key (stats/standardize (get dataset key))))
          {}
          (keys dataset)))

(def boston-std
  (-> (standardize-column boston)
      ds/dataset
      (ds/add-columns {:chas (:chas boston)})
                       ;:zn   (:zn boston)})
      (ds/reorder-columns regressors response)))

(ds/info boston-std)

;; ### Histogram
^kind/vega
(let [data (ds/rows boston-std :as-maps)
      column-names (ds/column-names boston)]
  {:data   {:values data}
   :repeat {:column column-names}
   :spec   {:mark     "bar"
            :encoding {:x {:field {:repeat "column"} :type "quantitative"}
                       :y {:aggregate "count"}}}})

;; ### Box plots
^kind/vega
(let [data (ds/rows boston-std :as-maps)
      column-names (ds/column-names boston)]
  {:data   {:values data}
   :repeat {:column column-names}
   :spec   {:width    60 :mark "boxplot"
            :encoding {:y {:field {:repeat "column"} :type "quantitative" :scale {:zero false}}}}})

;; #### Outliers
(let [columns (ds/column-names boston-std)]
  (->> (for [column columns]
         (vector column (count (stats/outliers (get boston-std column)))))
       (sort-by first)))

;; ### Pairs plot
^kind/vega
(let [data (ds/rows boston-std :as-maps)
      column-names (ds/column-names boston-std)]
  {:data   {:values data}
   :repeat {:column column-names
            :row    column-names}
   :spec   {:height   100 :width 100
            :mark     "circle"
            :encoding {:x {:field {:repeat "column"} :type "quantitative" :scale {:zero false}}
                       :y {:field {:repeat "row"} :type "quantitative" :scale {:zero false}}}}})

(let [combos (combo/combinations regressors 2)]
  (for [[x y] combos]
    (assoc {} [x y] (stats/correlation (get boston-std x) (get boston-std y)))))

(for [[x y] (mapv (fn [r] [response r]) regressors)]
  (assoc {} [x y] (stats/correlation (get boston-std x) (get boston-std y))))

;; ## Tukey's Ladder Transformation
(defn box-cox-transform [data lambda]
  (map #(if (== lambda 0.0)
          (Math/log %)
          (/ (- (Math/pow (+ % 1) lambda) 1) lambda)) data))

(defn skewness-diff [data lambda]
  (let [transformed-data (box-cox-transform data lambda)]
    (Math/abs (stats/skewness transformed-data))))

(defn find-optimal-lambda [data start-lambda end-lambda step]
  (let [lambdas (range start-lambda (+ end-lambda step) step)
        skewnesses (map #(skewness-diff data %) lambdas)
        paired (map vector lambdas skewnesses)
        sorted (sort-by second paired)]
    (first (first sorted))))

(defn box-cox-optimal [data]
  (let [optimal-lambda (find-optimal-lambda data -2 5 0.5)]
    (box-cox-transform data optimal-lambda)))

(defn apply-find-optimal-lambda [dataset]
  (let [columns (keys dataset)]
    (map vector columns
         (map #(find-optimal-lambda (get dataset %) -2 5 0.5) columns))))

(apply-find-optimal-lambda boston)

(defn apply-box-cox-to-dataset [dataset]
  (reduce (fn [acc key]
            (assoc acc key (box-cox-optimal (get dataset key))))
          {}
          (keys dataset)))

(def boston-transformed
  (-> (apply-box-cox-to-dataset boston)
      ds/dataset
      (ds/add-columns {:chas (:chas boston)})
      (ds/reorder-columns regressors response)))

(ds/info boston-transformed)

;; ### Histogram
^kind/vega
(let [data (ds/rows boston-transformed :as-maps)
      column-names (ds/column-names boston)]
  {:data   {:values data}
   :repeat {:column column-names}
   :spec   {:mark     "bar"
            :encoding {:x {:field {:repeat "column"} :type "quantitative"}
                       :y {:aggregate "count"}}}})

;; ### Box plots
^kind/vega
(let [data (ds/rows boston-transformed :as-maps)
      column-names (ds/column-names boston)]
  {:data   {:values data}
   :repeat {:column column-names}
   :spec   {:width    60 :mark "boxplot"
            :encoding {:y {:field {:repeat "column"} :type "quantitative" :scale {:zero false}}}}})

;; #### Outliers
(let [columns (ds/column-names boston-transformed)]
  (->> (for [column columns]
         (vector column (count (stats/outliers (get boston-transformed column)))))
       (sort-by first)))

;; ### Pairs plot
^kind/vega
(let [data (ds/rows boston-transformed :as-maps)
      column-names (ds/column-names boston-transformed)]
  {:data   {:values data}
   :repeat {:column column-names
            :row    column-names}
   :spec   {:height   100 :width 100
            :mark     "circle"
            :encoding {:x {:field {:repeat "column"} :type "quantitative" :scale {:zero false}}
                       :y {:field {:repeat "row"} :type "quantitative" :scale {:zero false}}}}})

(let [combos (combo/combinations regressors 2)]
  (for [[x y] combos]
    (assoc {} [x y] (stats/correlation (get boston-transformed x) (get boston-transformed y)))))

(for [[x y] (mapv (fn [r] [response r]) regressors)]
  (assoc {} [x y] (stats/correlation (get boston-transformed x) (get boston-transformed y))))
