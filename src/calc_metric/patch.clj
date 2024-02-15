(ns calc-metric.patch
  (:require [scicloj.metamorph.ml]))

;;start patching stuff at runtime.
(in-ns 'scicloj.metamorph.ml)

(defn- calc-metric [pipeline-fn metric-fn train-ds test-ds tune-options]
  (try
    (let [
          start-fit (System/nanoTime)                       ;;changed
          fitted-ctx (pipeline-fn {:metamorph/mode :fit :metamorph/data train-ds})
          end-fit (System/nanoTime)                         ;;changed

          eval-pipe-result-train (eval-pipe pipeline-fn fitted-ctx metric-fn train-ds (:other-metrices tune-options))
          eval-pipe-result-test (if (-> fitted-ctx :model ::unsupervised?)
                                  {:other-metrices []
                                   :timing         0
                                   :ctx            fitted-ctx
                                   :metric         0}
                                  (eval-pipe pipeline-fn fitted-ctx metric-fn test-ds (:other-metrices tune-options)))]

      {:fit-ctx         fitted-ctx
       :timing-fit      (- end-fit start-fit) ; (* (- end-fit start-fit) 1000000.0) ;;changed
       :train-transform eval-pipe-result-train
       :test-transform  eval-pipe-result-test})))

;;done patching
(in-ns 'calc-metric.patch)
