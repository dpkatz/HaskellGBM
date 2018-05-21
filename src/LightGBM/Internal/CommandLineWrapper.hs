{-# LANGUAGE OverloadedStrings #-}

module LightGBM.Internal.CommandLineWrapper
  ( run
  , OutLog(..)
  , ErrLog(..)
  ) where

import qualified Data.ByteString.Lazy as BSL
import           Data.Char (toLower)
import qualified Data.HashMap.Strict as M
import           Data.List (intercalate)
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Formatting as F
import           Refined (unrefine)
import           System.Exit (ExitCode(..))
import qualified System.Process.Typed as S

import qualified LightGBM.Parameters as P

-- Maps from values to relevant strings
taskPMap :: M.HashMap P.TaskType String
taskPMap =
  M.fromList
    [ (P.Train, "train")
    , (P.Predict, "predict")
    , (P.ConvertModel, "convert_model")
    , (P.Refit, "refit")
    ]

boosterPMap :: M.HashMap P.Booster String
boosterPMap =
  M.fromList [(P.GBDT, "gbdt"), (P.RandomForest, "rf"), (P.GOSS, "goss")]

mkGPUOption :: P.GPUParam -> String
mkGPUOption p =
  case p of
    P.GpuPlatformId platId -> "gpu_platform_id=" ++ show platId
    P.GpuDeviceId devId -> "gpu_device_id=" ++ show devId
    P.GpuUseDP tOrF -> "gpu_use_dp=" ++ show tOrF

mkParaOptions :: P.ParallelismParams -> [String]
mkParaOptions (P.MPIVer n) = ["num_machines=" ++ show n]
mkParaOptions (P.SocketVer n f p t) =
  [ "num_machines=" ++ show n
  , "machine_list_file=" ++ f
  , "local_listen_port=" ++ show p
  , "time_out=" ++ show t
  ]

directionPMap :: M.HashMap P.Direction String
directionPMap =
  M.fromList [(P.Increasing, "1"), (P.Decreasing, "-1"), (P.NoConstraint, "0")]

verbosityPMap :: M.HashMap P.VerbosityLevel String
verbosityPMap = M.fromList [(P.Fatal, "-1"), (P.Warn, "0"), (P.Info, "1")]

metricPMap :: M.HashMap P.Metric String
metricPMap =
  M.fromList
    [ (P.MeanAbsoluteError, "l1")
    , (P.MeanSquareError, "l2")
    , (P.L2_root, "l2root")
    , (P.QuantileRegression, "quantile")
    , (P.MAPELoss, "mape")
    , (P.HuberLoss, "huber")
    , (P.FairLoss, "fair")
    , (P.PoissonNegLogLikelihood, "poisson")
    , (P.GammaNegLogLikelihood, "gamma")
    , (P.GammaDeviance, "gamma_deviance")
    , (P.TweedieNegLogLiklihood, "tweedie")
    , (P.MAP, "map")
    , (P.AUC, "auc")
    , (P.BinaryLogloss, "binary_logloss")
    , (P.BinaryError, "binary_error")
    , (P.MultiLogloss, "multi_logloss")
    , (P.MultiError, "multi_error")
    , (P.Xentropy, "xentropy")
    , (P.XentLambda, "xentlambda")
    , (P.KullbackLeibler, "kldiv")
    ]

modelLangPMap :: M.HashMap P.ModelLang String
modelLangPMap = M.fromList [(P.CPP, "cpp")]

applicationPMap :: M.HashMap P.Application String
applicationPMap =
  M.fromList
    [ (P.Regression P.L1, "regression_l1")
    , (P.Regression P.L2, "regression_l2")
    , (P.Regression P.Huber, "huber")
    , (P.Regression P.Fair, "fair")
    , (P.Regression P.Poisson, "poisson")
    , (P.Regression P.Quantile, "quantile")
    , (P.Regression P.MAPE, "mape")
    , (P.Regression P.Gamma, "gamma")
    , (P.Binary, "binary")
    , (P.CrossEntropy P.XEntropy, "xentropy")
    , (P.CrossEntropy P.XEntropyLambda, "xentlambda")
    , (P.LambdaRank, "lambdarank")
    ]

mkTweedieString :: P.TweedieRegressionParam -> String
mkTweedieString (P.TweedieVariancePower p) = "tweedie_variance_power=" ++ show p

mkDartString :: P.DARTParam -> String
mkDartString (P.DropRate r) = "drop_rate=" ++ show r
mkDartString (P.SkipDrop r) = "skip_drop=" ++ show r
mkDartString (P.MaxDrop r) = "max_drop=" ++ show r
mkDartString (P.UniformDrop b) = "uniform_drop=" ++ show b
mkDartString (P.XGBoostDARTMode b) = "xgboost_dart_mode=" ++ show b
mkDartString (P.DropSeed b) = "drop_seed=" ++ show b

-- | Construct the option string for the command.
mkOptionString :: P.Param -> [String]
mkOptionString (P.ConfigFile f) = ["config=" ++ show f]
mkOptionString (P.Task t) = ["task=" ++ (taskPMap M.! t)]
mkOptionString (P.App (P.MultiClass P.MultiClassSimple n)) =
  ["application=multiclass", "num_classes=" ++ show n]
mkOptionString (P.App (P.MultiClass P.MultiClassOneVsAll n)) =
  ["application=multiclassova", "num_classes=" ++ show n]
mkOptionString (P.App (P.Regression (P.Tweedie tparams))) =
  ["application=tweedie"] ++ map mkTweedieString tparams
mkOptionString (P.App a) = ["application=" ++ (applicationPMap M.! a)]
mkOptionString (P.BoostingType (P.DART dartParams)) =
  ["boosting=dart"] ++ map mkDartString dartParams
mkOptionString (P.BoostingType b) = ["boosting=" ++ (boosterPMap M.! b)]
mkOptionString (P.TrainingData f) = ["data=" ++ show f]
mkOptionString (P.ValidationData f) = ["valid=" ++ show f]
mkOptionString (P.PredictionData f) = ["data=" ++ show f]
mkOptionString (P.Iterations n) = ["num_iterations=" ++ show n]
mkOptionString (P.LearningRate d) = ["learning_rate=" ++ show d]
mkOptionString (P.NumLeaves n) = ["num_leaves=" ++ show n]
mkOptionString (P.Parallelism P.Serial) = ["tree_learner=serial"]
mkOptionString (P.Parallelism (P.FeaturePar params)) =
  "tree_learner=feature" : mkParaOptions params
mkOptionString (P.Parallelism (P.DataPar params)) =
  "tree_learner=data" : mkParaOptions params
mkOptionString (P.Parallelism (P.VotingPar params)) =
  "tree_learner=voting" : mkParaOptions params
mkOptionString (P.NumThreads n) = ["num_threads=" ++ show n]
mkOptionString (P.Device P.CPU) = ["device=cpu"]
mkOptionString (P.Device (P.GPU gpuParams)) =
  "device=gpu" : map mkGPUOption gpuParams
mkOptionString (P.MaxDepth n) = ["max_depth=" ++ show n]
mkOptionString (P.MinDataInLeaf n) = ["min_data_in_leaf=" ++ show n]
mkOptionString (P.MinSumHessianInLeaf d) =
  ["min_sum_hessian_in_leaf=" ++ show d]
mkOptionString (P.FeatureFraction f) =
  ["feature_fraction=" ++ show (unrefine f)]
mkOptionString (P.FeatureFractionSeed s) = ["feature_fraction_seed=" ++ show s]
mkOptionString (P.BaggingFraction f) =
  ["bagging_fraction=" ++ show (unrefine f)]
mkOptionString (P.BaggingFreq n) = ["bagging_freq=" ++ show (unrefine n)]
mkOptionString (P.BaggingFractionSeed n) = ["bagging_seed=" ++ show n]
mkOptionString (P.EarlyStoppingRounds r) =
  ["early_stopping_round=" ++ show (unrefine r)]
mkOptionString (P.Regularization_L1 d) = ["lambda_l1=" ++ show d]
mkOptionString (P.Regularization_L2 d) = ["lambda_l2=" ++ show d]
mkOptionString (P.MaxDeltaStep s) = ["max_delta_step=" ++ show s]
mkOptionString (P.MinSplitGain sg) = ["min_split_gain=" ++ show sg]
mkOptionString (P.TopRate b) = ["top_rate=" ++ show b]
mkOptionString (P.OtherRate b) = ["other_rate=" ++ show b]
mkOptionString (P.MinDataPerGroup b) = ["min_data_per_group=" ++ show b]
mkOptionString (P.MaxCatThreshold b) = ["max_cat_threshold=" ++ show b]
mkOptionString (P.CatSmooth b) = ["cat_smooth=" ++ show b]
mkOptionString (P.CatL2 b) = ["cat_l2=" ++ show b]
mkOptionString (P.MaxCatToOneHot b) = ["max_cat_to_onehot=" ++ show b]
mkOptionString (P.TopK b) = ["top_k=" ++ show b]
mkOptionString (P.MonotoneConstraint cs) =
  ["monotone_constraint=" ++ intercalate "," (map (directionPMap M.!) cs)]
mkOptionString (P.MaxBin n) = ["max_bin=" ++ show (unrefine n)]
mkOptionString (P.MinDataInBin n) = ["min_data_in_bin=" ++ show n]
mkOptionString (P.DataRandomSeed i) = ["data_random_seed=" ++ show i]
mkOptionString (P.OutputModel f) = ["output_model=" ++ show f]
mkOptionString (P.InputModel f) = ["input_model=" ++ show f]
mkOptionString (P.OutputResult f) = ["output_result=" ++ show f]
mkOptionString (P.PrePartition b) = ["pre_partition=" ++ show b]
mkOptionString (P.IsSparse b) = ["is_sparse=" ++ show b]
mkOptionString (P.TwoRoundLoading b) = ["two_round=" ++ show b]
mkOptionString (P.SaveBinary b) = ["save_binary=" ++ show b]
mkOptionString (P.Verbosity v) = ["verbosity=" ++ verbosityPMap M.! v]
mkOptionString (P.HasHeader b) = ["header=" ++ show b]
mkOptionString (P.LabelColumn c) = ["label=" ++ P.colSelArgument c]
mkOptionString (P.WeightColumn c) = ["weight=" ++ P.colSelArgument c]
mkOptionString (P.QueryColumn c) = ["query=" ++ P.colSelArgument c]
mkOptionString (P.IgnoreColumns cs) =
  ["ignore_column=" ++ intercalate "," (map P.colSelArgument cs)]
mkOptionString (P.CategoricalFeatures cs) =
  ["categorical_feature=" ++ intercalate "," (map P.colSelArgument cs)]
mkOptionString (P.PredictRawScore b) = ["predict_raw_score=" ++ show b]
mkOptionString (P.PredictLeafIndex b) = ["predict_leaf_index=" ++ show b]
mkOptionString (P.PredictContrib b) = ["predict_contrib=" ++ show b]
mkOptionString (P.BinConstructSampleCount n) =
  ["bin_construct_sample_cnt=" ++ show n]
mkOptionString (P.NumIterationsPredict n) =
  ["num_iterations_predict=" ++ show n]
mkOptionString (P.PredEarlyStop b) = ["pred_early_stop=" ++ show b]
mkOptionString (P.PredEarlyStopFreq n) = ["pred_early_stop_freq=" ++ show n]
mkOptionString (P.PredEarlyStopMargin d) = ["pred_early_stop_margin=" ++ show d]
mkOptionString (P.UseMissing b) = ["use_missing=" ++ show b]
mkOptionString (P.ZeroAsMissing b) = ["zero_as_missing=" ++ show b]
mkOptionString (P.InitScoreFile f) = ["init_score_file=" ++ f]
mkOptionString (P.ValidInitScoreFile f) =
  ["valid_init_score_file=" ++ intercalate "," f]
mkOptionString (P.ForcedSplits f) = ["forced_splits=" ++ f]
mkOptionString (P.Sigmoid d) = ["sigmoid=" ++ show d]
mkOptionString (P.Alpha d) = ["alpha=" ++ show d]
mkOptionString (P.FairC d) = ["fair_c=" ++ show d]
mkOptionString (P.PoissonMaxDeltaStep d) = ["poisson_max_delta_step=" ++ show d]
mkOptionString (P.ScalePosWeight d) = ["scale_pos_weight=" ++ show d]
mkOptionString (P.BoostFromAverage b) = ["boost_from_average=" ++ show b]
mkOptionString (P.IsUnbalance b) = ["is_unbalance=" ++ show b]
mkOptionString (P.MaxPosition n) = ["max_position=" ++ show n]
mkOptionString (P.LabelGain ds) =
  ["label_gain=" ++ intercalate "," (map show ds)]
mkOptionString (P.RegSqrt b) = ["reg_sqrt=" ++ show b]
mkOptionString (P.Metric ms) =
  let metrics = ["metric=" ++ intercalate "," (map metricName ms)]
   in case ndcgEvalPts ms of
        [] -> metrics
        pts -> metrics ++ pts
  where
    metricName (P.NDCG _) = "ndcg"
    metricName m = metricPMap M.! m
    ndcgEvalPts [] = []
    ndcgEvalPts (P.NDCG (Just pts):_) =
      ["ndcg_at=" ++ intercalate "," (map show pts)]
    ndcgEvalPts (_:mets) = ndcgEvalPts mets
mkOptionString (P.MetricFreq f) = ["metric_freq=" ++ show (unrefine f)]
mkOptionString (P.TrainingMetric b) =
  ["training_metric=" ++ fmap toLower (show b)]
mkOptionString (P.ConvertModelLanguage l) =
  ["convert_model_language=" ++ (modelLangPMap M.! l)]
mkOptionString (P.ConvertModelOutput f) = ["convert_model=" ++ f]

-- | A transcript of the stdout output of running LightGBM
newtype OutLog = OutLog T.Text deriving Show

-- | A transcript of the stderr output of running LightGBM
newtype ErrLog = ErrLog T.Text deriving Show

-- | Run the LightGBM executable with appropriate parameters
run ::
     FilePath -- ^ The path to the lightgbm executable
  -> [P.Param] -- ^ A list of parameters to override defaults
  -> IO (Either ErrLog OutLog)
run executable params = do
  let optStrings = concatMap mkOptionString params
      lgbmProc = S.proc executable optStrings
  (exitcode, out, err) <-
    S.readProcess (S.setStdin (S.byteStringInput "") lgbmProc)
  case exitcode of
    ExitSuccess -> return $ Right . OutLog . TE.decodeUtf8 . BSL.toStrict $ out
    ExitFailure code -> do
      let reason = F.sformat ("lightGBM failed with code : " F.% F.int) code
          errlog = TE.decodeUtf8 . BSL.toStrict $ err
      return $ Left . ErrLog . T.unlines $ [reason, errlog]
