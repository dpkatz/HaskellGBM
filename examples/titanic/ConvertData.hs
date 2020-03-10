-- |

{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module ConvertData (csvFilter, testFilter, predsToKaggleFormat) where

import           Control.Exception (IOException)
import qualified Control.Exception as Exception
import qualified Data.ByteString.Lazy as BSL
import           Data.Csv ((.:), (.=))
import qualified Data.Csv as CSV
import qualified Data.Foldable as Foldable
import qualified Data.Text as T
import qualified Data.Text.Encoding as TE
import qualified Data.Vector as V
import           System.IO (Handle)

import           LightGBM.Utils.Csv (readColumn)

-- Parsing the Titanic data
data Sex
  = Male
  | Female
  deriving (Eq, Show)

data Embarkation
  = Cherbourg
  | Queenstown
  | Southampton
  | Other T.Text
  deriving (Eq, Show)

newtype Survived =
  Survived Bool
  deriving (Eq, Show, Ord)

data PassengerData = PD
  { passengerID :: Int
  , passengerClass :: Int
  , name :: String
  , sex :: Sex
  , age :: Maybe Double
  , siblingsAndSpouses :: Int
  , parentsAndChildren :: Int
  , ticket :: String
  , fare :: Maybe Double
  , cabin :: Maybe String
  , embarkationPoint :: Embarkation
  } deriving (Eq, Show)

data PassengerRecord = PR
  { passengerData :: PassengerData
  , survived :: Survived
  } deriving (Eq, Show)

instance CSV.FromField Survived where
  parseField "0" = pure $ Survived False
  parseField _ = pure $ Survived True

instance CSV.ToField Survived where
  toField (Survived True) = "1"
  toField (Survived False) = "0"

instance CSV.FromField Sex where
  parseField val =
    let lower = T.toLower . TE.decodeUtf8 $ val
     in case lower of
          "female" -> pure Female
          "male" -> pure Male
          other -> error ("Incorrect sex detected:  " ++ show other)

instance CSV.ToField Sex where
  toField Male = "1"
  toField Female = "0"

instance CSV.FromField Embarkation where
  parseField "S" = pure Southampton
  parseField "C" = pure Cherbourg
  parseField "Q" = pure Queenstown
  parseField other = pure $ Other (TE.decodeUtf8 other)

instance CSV.ToField Embarkation where
  toField Cherbourg = "0"
  toField Queenstown = "1"
  toField Southampton = "2"
  toField (Other _) = "3"

instance CSV.FromNamedRecord PassengerData where
  parseNamedRecord r =
    PD
    <$> r .: "PassengerId"
    <*> r .: "Pclass"
    <*> r .: "Name"
    <*> r .: "Sex"
    <*> r .: "Age"
    <*> r .: "SibSp"
    <*> r .: "Parch"
    <*> r .: "Ticket"
    <*> r .: "Fare"
    <*> r .: "Cabin"
    <*> r .: "Embarked"

instance CSV.FromNamedRecord PassengerRecord where
  parseNamedRecord r = PR <$> CSV.parseNamedRecord r <*> r .: "Survived"

-- | When writing the output CSV, we want to strip out irrelevant
-- features (like name and cabin) and convert the categorical features
-- into integral values (which is what LightGBM requires at the
-- moment).
instance CSV.ToNamedRecord PassengerData where
  toNamedRecord PD {..} =
    CSV.namedRecord
    [ "PassengerId" .= passengerID
    , "Pclass" .= passengerClass
    , "Sex" .= sex
    , "Age" .= age
    , "SibSp" .= siblingsAndSpouses
    , "Parch" .= parentsAndChildren
    , "Fare" .= fare
    , "Embarked" .= embarkationPoint
    ]

instance CSV.ToNamedRecord PassengerRecord where
  toNamedRecord PR {..} =
    CSV.namedRecord ["Survived" .= survived] <> CSV.toNamedRecord passengerData

instance CSV.DefaultOrdered PassengerData where
  headerOrder _ =
    CSV.header
      ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

instance CSV.DefaultOrdered PassengerRecord where
  headerOrder _ =
    CSV.header
      ["Survived", "PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

encodeItems ::
     (Foldable t, CSV.DefaultOrdered a, CSV.ToNamedRecord a)
  => t a
  -> BSL.ByteString
encodeItems = CSV.encodeDefaultOrderedByName . Foldable.toList

encodeFilteredRecordsToFile ::
     (Foldable t, CSV.DefaultOrdered a, CSV.ToNamedRecord a)
  => Handle
  -> t a
  -> IO (Either String ())
encodeFilteredRecordsToFile outHandle =
  catchShowIO . BSL.hPut outHandle . encodeItems

catchShowIO :: IO a -> IO (Either String a)
catchShowIO action =
  fmap Right action
    `Exception.catch` handleIOException
  where
    handleIOException
      :: IOException
      -> IO (Either String a)
    handleIOException =
      return . Left . show

importNamedCSV :: CSV.FromNamedRecord a => FilePath -> IO [a]
importNamedCSV filename = do
  bytes <- BSL.readFile filename
  case CSV.decodeByName bytes of
    Left err -> do putStrLn err
                   return []
    Right drv -> return . V.toList . snd $ drv

csvFilter :: FilePath -> Handle -> IO (Either String ())
csvFilter initial final = do
  prs <- V.fromList <$> (importNamedCSV initial :: IO [PassengerRecord])
  encodeFilteredRecordsToFile final prs

testFilter :: FilePath -> Handle -> IO (Either String ())
testFilter initial final = do
  prs <- V.fromList <$> importNamedCSV initial :: IO (V.Vector PassengerData)
  encodeFilteredRecordsToFile final prs



-- Output to the Kaggle-mandated format for the Titanic competition

data KaggleTitanicRecord = KTR
  { outPassengerId :: Int
  , outSurvived :: Int
  } deriving (Eq, Show)

instance CSV.ToNamedRecord KaggleTitanicRecord where
  toNamedRecord KTR {..} =
    CSV.namedRecord ["PassengerId" .= outPassengerId, "Survived" .= outSurvived]

predsToKaggleFormat :: BSL.ByteString -> BSL.ByteString -> BSL.ByteString
predsToKaggleFormat testData predictions =
  let passengerIds = readColumn 0 CSV.HasHeader testData :: V.Vector Int
      preds =
        V.map round (readColumn 0 CSV.NoHeader predictions :: V.Vector Double)
      outdata = V.toList $ V.zipWith KTR passengerIds preds
      header = CSV.header ["PassengerId", "Survived"]
   in CSV.encodeByName header outdata
