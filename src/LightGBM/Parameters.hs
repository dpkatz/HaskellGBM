-- | Parameter types for LightGBM
{-# LANGUAGE DeriveGeneric #-}

module LightGBM.Parameters
  ( -- * Parameters
    Application(..)
  , Booster(..)
  , DARTParam(..)
  , Device(..)
  , Direction(..)
  , GPUParam(..)
  , LocalListenPort
  , MachineListFile
  , Metric(..)
  , Minutes
  , ModelLang(..)
  , MultiClassStyle(..)
  , NDCGEvalPositions
  , NumClasses
  , NumMachines
  , ParallelismParams(..)
  , ParallelismStyle(..)
  , Param(..)
  , RegressionApp(..)
  , TaskType(..)
  , TweedieRegressionParam(..)
  , VerbosityLevel(..)
  , XEApp(..)
  -- * Utilities
  , ColumnSelector(..)
  , colSelArgument
  ) where

import Data.Hashable (Hashable)
import GHC.Generics (Generic)
import Numeric.Natural (Natural)

import LightGBM.Utils (OneToTwoLeftSemiClosed, PositiveInt, ProperFraction)

-- | Parameters control the behavior of lightGBM.
data Param
  = ConfigFile FilePath -- ^ Path to config file
  | Task TaskType -- ^ Task to perform
  | App Application -- ^ Application
  | BoostingType Booster -- ^ Booster to apply - by default is 'GBDT'
  | TrainingData FilePath -- ^ Path to training data
  | ValidationData FilePath -- ^ Path to testing data
  | PredictionData FilePath -- ^ Path to data to use for a prediction
  | Iterations Natural
  | LearningRate Double
  | NumLeaves Natural
  | Parallelism ParallelismStyle
  | NumThreads Natural
  | Device Device
  | MaxDepth Natural
  | MinDataInLeaf Natural
  | MinSumHessianInLeaf Double
  | FeatureFraction ProperFraction
  | FeatureFractionSeed Int
  | BaggingFraction ProperFraction
  | BaggingFreq PositiveInt
  | BaggingFractionSeed Int
  | EarlyStoppingRounds PositiveInt
  | Regularization_L1 Double
  | Regularization_L2 Double
  | MaxDeltaStep Double
  | MinSplitGain Double
  | TopRate Double -- ^ GOSS only
  | OtherRate Double -- ^ GOSS only
  | MinDataPerGroup Natural -- ^ Minimum number of data points per categorial group
  | MaxCatThreshold Natural
  | CatSmooth Double
  | CatL2 Double -- ^ L2 regularization in categorical split
  | MaxCatToOneHot PositiveInt
  | TopK Natural -- ^ VotingPar only
  | MonotoneConstraint [Direction] -- ^ Length of directions = number of features
  | MaxBin PositiveInt
  | MinDataInBin Natural
  | DataRandomSeed Int
  | OutputModel FilePath -- ^ Where to persist the model after training
  | InputModel FilePath -- ^ Filepath to a persisted model to use for prediction or additional training
  | OutputResult FilePath -- ^ Where to persist the output of a prediction task
  | PrePartition Bool
  | IsSparse Bool
  | TwoRoundLoading Bool
  | SaveBinary Bool
  | Verbosity VerbosityLevel
  | HasHeader Bool -- ^ True if the input data has a header
  | LabelColumn ColumnSelector -- ^ Which column has the labels
  | WeightColumn ColumnSelector -- ^ Which column has the weights
  | QueryColumn ColumnSelector
  | IgnoreColumns [ColumnSelector] -- ^ Select columns to ignore in training
  | CategoricalFeatures [ColumnSelector] -- ^ Select columns to use as features
  | PredictRawScore Bool -- ^ Prediction Only; true = raw scores only, false = transformed scores
  | PredictLeafIndex Bool -- ^ Prediction Only
  | PredictContrib Bool -- ^ Prediction Only
  | BinConstructSampleCount Natural
  | NumIterationsPredict Natural -- ^ Prediction Only; how many trained predictions
  | PredEarlyStop Bool
  | PredEarlyStopFreq Natural
  | PredEarlyStopMargin Double
  | UseMissing Bool
  | ZeroAsMissing Bool
  | InitScoreFile FilePath
  | ValidInitScoreFile [FilePath]
  | ForcedSplits FilePath
  | Sigmoid Double -- ^ Used in Binary classification and LambdaRank
  | Alpha Double -- ^ Used in Huber loss and Quantile regression
  | FairC Double -- ^ Used in Fair loss
  | PoissonMaxDeltaStep Double -- ^ Used in Poisson regression
  | ScalePosWeight Double -- ^ Used in Binary classification
  | BoostFromAverage Bool -- ^ Used only in RegressionL2 task
  | IsUnbalance Bool -- ^ Used in Binary classification (set to true if training data are unbalanced)
  | MaxPosition Natural -- ^ Used in LambdaRank
  | LabelGain [Double] -- ^ Used in LambdaRank
  | RegSqrt Bool -- ^ Only used in RegressionL2
  | Metric [Metric] -- ^ Loss Metric
  | MetricFreq PositiveInt
  | TrainingMetric Bool
  | ConvertModelLanguage ModelLang
  | ConvertModelOutput FilePath
  deriving (Eq, Show)

-- | LightGBM supports various tasks:
data TaskType
  = Train -- ^ Training
  | Predict -- ^ Prediction
  | ConvertModel -- ^ Conversion into an if-then-else format
  | Refit -- ^ Refitting existing models with new data
  deriving (Eq, Show, Generic)
instance Hashable TaskType

-- | Different types of Boosting approaches
data Booster
  = GBDT -- ^ Gradient Boosting Decision Tree
  | RandomForest
  | DART [DARTParam] -- ^ Dropouts meet Multiple Additive Regression Trees
  | GOSS -- ^ Gradient-based One-Sided Sampling
  deriving (Eq, Show, Generic)
instance Hashable Booster

data TweedieRegressionParam =
  TweedieVariancePower OneToTwoLeftSemiClosed -- ^ Control Tweedie variance in range [1, 2) - 1 is like Poisson, 2 is like Gamma
  deriving (Eq, Show, Generic)
instance Hashable TweedieRegressionParam

-- | Different types of regression metrics
data RegressionApp
  = L1 -- ^ Absolute error metric
  | L2 -- ^ RMS errror metric
  | Huber
  | Fair
  | Poisson
  | Quantile
  | MAPE
  | Gamma
  | Tweedie [TweedieRegressionParam]
  deriving (Eq, Show, Generic)
instance Hashable RegressionApp

-- | Multi-classification styles
data MultiClassStyle
  = MultiClassSimple
  | MultiClassOneVsAll
  deriving (Eq, Show, Generic)
instance Hashable MultiClassStyle

data XEApp
  = XEntropy
  | XEntropyLambda
  deriving (Eq, Show, Generic)
instance Hashable XEApp

type NumMachines = PositiveInt

type MachineListFile = FilePath

type LocalListenPort = Natural

type Minutes = Natural

data ParallelismParams
  = SocketVer { nMachines :: NumMachines
              , machineList :: MachineListFile
              , port :: LocalListenPort
              , timeOut :: Minutes }
  | MPIVer { nMachines :: NumMachines }
  deriving (Eq, Show, Generic)
instance Hashable ParallelismParams

data ParallelismStyle
  = Serial
  | FeaturePar ParallelismParams
  | DataPar ParallelismParams
  | VotingPar ParallelismParams
  deriving (Eq, Show, Generic)
instance Hashable ParallelismStyle

data GPUParam
  = GpuPlatformId Int
  | GpuDeviceId Int
  | GpuUseDP Bool
  deriving (Eq, Show, Generic)
instance Hashable GPUParam

data Device
  = CPU
  | GPU [GPUParam]
  deriving (Eq, Show, Generic)
instance Hashable Device

data Direction
  = Increasing
  | Decreasing
  | NoConstraint
  deriving (Eq, Show, Generic)
instance Hashable Direction

data VerbosityLevel
  = Fatal
  | Warn
  | Info
  deriving (Eq, Show, Generic)
instance Hashable VerbosityLevel

type NDCGEvalPositions = [Natural]

data Metric
  = MeanAbsoluteError -- L1
  | MeanSquareError -- L2
  | L2_root
  | QuantileRegression
  | MAPELoss
  | HuberLoss
  | FairLoss
  | PoissonNegLogLikelihood
  | GammaNegLogLikelihood
  | GammaDeviance
  | TweedieNegLogLiklihood
  | NDCG (Maybe NDCGEvalPositions)
  | MAP
  | AUC
  | BinaryLogloss
  | BinaryError
  | MultiLogloss
  | MultiError
  | Xentropy
  | XentLambda
  | KullbackLeibler
  deriving (Eq, Show, Generic)
instance Hashable Metric

data ModelLang =
  CPP
  deriving (Eq, Show, Generic)
instance Hashable ModelLang

type NumClasses = Natural

-- | LightGBM can be used for a variety of applications
data Application
  = Regression RegressionApp -- ^ Regression
  | Binary -- ^ Binary classification
  | MultiClass MultiClassStyle NumClasses -- ^ Multi-class
  | CrossEntropy XEApp
  | LambdaRank
  deriving (Eq, Show, Generic)
instance Hashable Application

-- | Parameters exclusively for the DART booster
data DARTParam
  = DropRate Double
  | SkipDrop Double
  | MaxDrop PositiveInt
  | UniformDrop Bool
  | XGBoostDARTMode Bool
  | DropSeed Int
  deriving (Eq, Show, Generic)
instance Hashable DARTParam

-- | Some parameters are based on column selection either by index or
-- by name.  A 'ColumnSelector' encapsulates this flexibility.
data ColumnSelector
  = Index Natural
  | ColName String
  deriving (Eq, Show)

colSelArgument :: ColumnSelector -> String
colSelArgument (Index i) = show i
colSelArgument (ColName s) = s
