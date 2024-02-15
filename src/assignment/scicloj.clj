(ns assignment.scicloj
  (:require
    [assignment.eda :refer [boston boston-transformed]]
    [calc-metric.patch]
    [fastmath.stats :as stats]
    [scicloj.ml.core :as ml]
    [scicloj.ml.dataset :as ds]
    [scicloj.ml.metamorph :as mm]))

;; # Clojure with Smile Algorithms
;; ## Define regressors and response
(def response :medv)
(def regressors
  (ds/column-names boston (complement #{response})))

;; ## Convert Boston column types
(ds/info boston)

;; Right now, boston has too informative columns, viz. type :float64. Normally, I'd prefer to have more information per entry, however, trying to run this notebook without converting to :float32 breaks the JVM.

(-> boston
    (ds/convert-types :type/float64 :float32)
    ds/info)

(def boston-32
  (-> boston
      (ds/convert-types :type/float64 :float32)))

;; ## Setup Pipelines
(def pipeline-fn
  (ml/pipeline
    (mm/set-inference-target response)))

;; ### Generic pipeline function
(defn create-model-pipeline
  [model-type params]
  (ml/pipeline
    pipeline-fn
    {:metamorph/id :model}
    (mm/model (merge {:model-type model-type} params))))

;; #### Ridge context
(defn ridge-pipe-fn
  [params]
  (create-model-pipeline :smile.regression/ridge params))

;; #### Lasso context
(defn lasso-pipe-fn
  [params]
  (create-model-pipeline :smile.regression/lasso params))

;; #### Best-subset context
;; Best subset has a different pattern/logic of pipelining. Whereas Ridge and Lasso have hyperparameters, the best subset algorithm, as I am implementing it, is nothing but ordinary least square models iterated over all combinations of regressors.

(defn all-combinations [coll]
  (letfn [(comb [coll]
            (if (empty? coll)
              [[]]
              (let [rest (comb (rest coll))]
                (concat rest (map #(cons (first coll) %) rest)))))]
    (rest (comb coll))))

(Math/pow 2 (count regressors))

;; Above, we see the number of combination possibilities with the number of regressors we have, 14. 2 to the power 13 is 8192.

;; Below, we see the count of the `all-combinations` function on the list of regressors. The number is 2 to the power 13 minus 1. The difference of 1 is that I did not include a null model, i.e. no regressors.

(count (all-combinations regressors))

(defn best-subset-pipe-fn
  [dataset y regrs]
  (let [combinations (all-combinations regrs)]
    (pmap (fn [Xs]                                          ; test `map` vs `pmap` vs `mapv`
            (ml/pipeline
              (mm/select-columns (cons y Xs))
              (mm/set-inference-target y)
              {:metamorph/id :model}
              (mm/model {:model-type :smile.regression/ordinary-least-square})))
          combinations)))

;; ## Pipeline Functions
;; ### Evaluate pipeline
(defn evaluate-pipe [pipe data]
  (ml/evaluate-pipelines
    pipe
    data
    stats/omega-sq
    :accuracy
    {:other-metrices                   [{:name :mae :metric-fn ml/mae}
                                        {:name :rmse :metric-fn ml/rmse}]
     :return-best-pipeline-only        false
     :return-best-crossvalidation-only true}))

;; ### Generate hyperparameters for models
(defn generate-hyperparams [model-type]
  (case model-type
    :ridge (ml/sobol-gridsearch
             (assoc-in (ml/hyperparameters :smile.regression/ridge) [:lambda :n-steps] 500))
    :lasso (take 500 (ml/sobol-gridsearch (ml/hyperparameters :smile.regression/lasso)))))

;; ### Evaluate a single model
(defn evaluate-model [dataset split-fn model-type model-fn]
  (let [data-split (split-fn dataset)
        pipelines (cond
                    (= model-type :best-subset)
                    (model-fn dataset response regressors)
                    :else (map model-fn (generate-hyperparams model-type)))]
    (evaluate-pipe pipelines data-split)))

;; ### Split functions
(defn train-test [dataset]
  (ds/split->seq dataset :bootstrap {:seed 123 :repeats 30}))

(defn train-val [dataset]
  (let [ds-split (train-test dataset)]
    (ds/split->seq (:train (first ds-split)) :kfold {:seed 123 :k 5})))

;; ### Define model types and corresponding functions as a vector of vectors
(def model-type-fns
  {:ridge ridge-pipe-fn
   :lasso lasso-pipe-fn})

;; ### Evaluate models for a dataset
(defn evaluate-models [dataset split-fn]
  (mapv (fn [[model-type model-fn]]
          (evaluate-model dataset split-fn model-type model-fn))
        model-type-fns))

;; ### Evaluate separately
(def ridge-lasso-models (evaluate-models boston-32 train-val))
(comment
  (def best-subset-model
    (evaluate-model boston-32 train-val :best-subset best-subset-pipe-fn)))

;; ## Extract Useable Models
(defn best-models [eval]
  (->> eval
       flatten
       (map
         #(hash-map :summary (ml/thaw-model (get-in % [:fit-ctx :model]))
                    :fit-ctx (:fit-ctx %)
                    :timing-fit (:timing-fit %)
                    :metric ((comp :metric :test-transform) %)
                    :other-metrices ((comp :other-metrices :test-transform) %)
                    :other-metric-1 ((comp :metric first) ((comp :other-metrices :test-transform) %))
                    :other-metric-2 ((comp :metric second) ((comp :other-metrices :test-transform) %))
                    :params ((comp :options :model :fit-ctx) %)
                    :pipe-fn (:pipe-fn %)))
       (sort-by :metric)))

(def best-val-ridge
  (-> (first ridge-lasso-models)
      best-models
      reverse))

(-> best-val-ridge first :summary)
(-> best-val-ridge first :metric)
(-> best-val-ridge first :other-metrices)
(-> best-val-ridge first :params)

(def best-val-lasso
  (-> (second ridge-lasso-models)
      best-models
      reverse))

(-> best-val-lasso first :summary)
(-> best-val-lasso first :metric)
(-> best-val-lasso first :other-metrices)
(-> best-val-lasso first :params)

(comment
  (def best-val-subset
    (-> best-subset-model
        best-models
        reverse))

  (-> best-val-subset first :summary)
    ;=>
    ;#object[smile.regression.LinearModel
    ;        0x7a721884
    ;        "Linear Model:
    ;
    ;       Residuals:
    ;              Min          1Q      Median          3Q         Max
    ;         -18.3055     -3.3507     -0.9813      2.3172     31.4060
    ;
    ;       Coefficients:
    ;                         Estimate Std. Error    t value   Pr(>|t|)
    ;       Intercept          -2.4171     4.7582    -0.5080     0.6118
    ;       crim               -0.0852     0.0381    -2.2372     0.0258 *
    ;       zn                 -0.0117     0.0161    -0.7239     0.4696
    ;       chas                3.7276     1.1892     3.1345     0.0018 **
    ;       rm                  6.5272     0.4104    15.9034     0.0000 ***
    ;       age                -0.0486     0.0134    -3.6262     0.0003 ***
    ;       ptratio            -0.9594     0.1583    -6.0608     0.0000 ***
    ;       b                   0.0149     0.0033     4.4582     0.0000 ***
    ;       ---------------------------------------------------------------------
    ;       Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ;
    ;       Residual standard error: 5.6289 on 397 degrees of freedom
    ;       Multiple R-squared: 0.6181,    Adjusted R-squared: 0.6113
    ;       F-statistic: 91.7731 on 8 and 397 DF,  p-value: 5.506e-79
    ;       "])
  (-> best-val-subset first :metric)
    ;=> 0.8303967218087499)
  (-> best-val-subset first :other-metrices)
    ;=>
    ;({:name :mae, :metric-fn #object[scicloj.ml.core$mae 0x798eacab "scicloj.ml.core$mae@798eacab"], :metric 2.770744533104}
    ; {:name :rmse,
    ;  :metric-fn #object[scicloj.ml.core$rmse 0x4350b8e7 "scicloj.ml.core$rmse@4350b8e7"],
    ;  :metric 3.7412435449760975}))
  (-> best-val-subset first :fit-ctx :model :feature-columns)
    ;=> [:crim :zn :chas :rm :age :ptratio :b]

  (def best-subset-regressors
    (-> best-val-subset first :fit-ctx :model :feature-columns)))

;; ## Build final models for evaluation
;; ### Ridge
(def final-model-ridge
  (-> (evaluate-pipe
        (map ridge-pipe-fn
             (-> best-val-ridge first :params))
        (train-test boston-32))
      last
      best-models))

(-> final-model-ridge first :summary)
(-> final-model-ridge first :metric)
(-> final-model-ridge first :other-metrices)
(-> final-model-ridge first :params)

;; ### Lasso
(def final-model-lasso
  (-> (evaluate-pipe
        (map lasso-pipe-fn
             (-> best-val-lasso first :params))
        (train-test boston-32))
      last
      best-models))

(-> final-model-lasso first :summary)
(-> final-model-lasso first :metric)
(-> final-model-lasso first :other-metrices)
(-> final-model-lasso first :params)

;; ### Best Subset
(defn pipeline-best-subset-fn [y Xs]
  (ml/pipeline
    (mm/select-columns (cons y Xs))
    (mm/set-inference-target y)))

(defn ols-pipe-fn [y Xs]
  (ml/pipeline
    (pipeline-best-subset-fn y Xs)
    {:metamorph/id :model}
    (mm/model {:model-type :smile.regression/ordinary-least-square})))

(comment
  (def final-best-subset
    (-> (evaluate-pipe
          [(ols-pipe-fn response best-subset-regressors)]
          (train-test boston-32))
        best-models))

 (-> final-best-subset first :summary)
   ;=>
   ;#object[smile.regression.LinearModel
   ;        0x9334757
   ;        "Linear Model:
   ;
   ;         Residuals:
   ;                Min          1Q      Median          3Q         Max
   ;           -11.7343     -3.2668     -0.9311      1.8266     38.1650
   ;
   ;         Coefficients:
   ;                           Estimate Std. Error    t value   Pr(>|t|)
   ;         Intercept          -2.6685     4.5855    -0.5819     0.5609
   ;         crim               -0.1507     0.0337    -4.4762     0.0000 ***
   ;         zn                  0.0059     0.0152     0.3913     0.6958
   ;         chas                3.7895     1.2637     2.9989     0.0028 **
   ;         rm                  5.9341     0.4373    13.5685     0.0000 ***
   ;         age                -0.0350     0.0126    -2.7764     0.0057 **
   ;         ptratio            -0.8448     0.1477    -5.7189     0.0000 ***
   ;         b                   0.0168     0.0030     5.6008     0.0000 ***
   ;         ---------------------------------------------------------------------
   ;         Significance codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
   ;
   ;         Residual standard error: 6.0676 on 498 degrees of freedom
   ;         Multiple R-squared: 0.5215,    Adjusted R-squared: 0.5148
   ;         F-statistic: 77.5462 on 8 and 498 DF,  p-value: 1.140e-75
   ;         "])
 (-> final-best-subset first :metric)
   ;=> 0.7419895970972805)
 (-> final-best-subset first :other-metrices)
   ;=>
   ;({:name :mae,
   ;  :metric-fn #object[scicloj.ml.core$mae 0x798eacab "scicloj.ml.core$mae@798eacab"],
   ;  :metric 3.266730104211804}
   ; {:name :rmse,
   ;  :metric-fn #object[scicloj.ml.core$rmse 0x4350b8e7 "scicloj.ml.core$rmse@4350b8e7"],
   ;  :metric 4.641450683867226}))
 (-> final-best-subset first :fit-ctx :model :feature-columns))
   ;=> [:crim :zn :chas :rm :age :ptratio :b])

;; # Transformation Data
(def boston-trans-32
  (-> boston-transformed
      (ds/convert-types :type/float64 :float32)))

;; ### Evaluate separately
(def ridge-lasso-trans-models (evaluate-models boston-trans-32 train-val))

(def best-val-trans-ridge
  (-> (first ridge-lasso-trans-models)
      best-models
      reverse))

(-> best-val-trans-ridge first :params)

(def best-val-trans-lasso
  (-> (second ridge-lasso-trans-models)
      best-models
      reverse))

(-> best-val-trans-lasso first :params)

;; ## Build final models for evaluation
;; ### Ridge
(def final-model-trans-ridge
  (-> (evaluate-pipe
        (map ridge-pipe-fn
             (-> best-val-trans-ridge first :params))
        (train-test boston-trans-32))
      last
      best-models))

(-> final-model-trans-ridge first :summary)
(-> final-model-trans-ridge first :metric)
(-> final-model-trans-ridge first :other-metrices)

;; ### Lasso
(def final-model-trans-lasso
  (-> (evaluate-pipe
        (map lasso-pipe-fn
             (-> best-val-trans-lasso first :params))
        (train-test boston-trans-32))
      last
      best-models))

(-> final-model-trans-lasso first :summary)
(-> final-model-trans-lasso first :metric)
(-> final-model-trans-lasso first :other-metrices)
