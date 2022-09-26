using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML;
using MachineLearning.Shared.Common;
using System.Data.SqlClient;

namespace MachineLearning.CaliforniaHosingPrice.Generator
{
    public class MLModelGenerator : BaseMLModelGenerator
    {
        public MLModelGenerator(MLContext mLContext) : base(mLContext, @"C:\MLData\CaliforniaHousingPriceModel.zip")
        {
        }

        public override IDataView CleanTrainDataView(IDataView trainDataView)
        {
            return mLContext.Data.FilterRowsByColumn(trainDataView, nameof(InputCaliforniaHosungPrice.MedianHouseValue), lowerBound: 50000, upperBound: 499999);
        }

        public override EstimatorChain<ColumnConcatenatingTransformer> GetDataProcessPipeline()
        {
            return mLContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(InputCaliforniaHosungPrice.MedianHouseValue))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "LatitudeEncoded", inputColumnName: nameof(InputCaliforniaHosungPrice.Latitude)))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "LongitudeEncoded", inputColumnName: nameof(InputCaliforniaHosungPrice.Longitude)))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "OceanProximityEncoded", inputColumnName: nameof(InputCaliforniaHosungPrice.OceanProximity)))
                .Append(mLContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(InputCaliforniaHosungPrice.HousingMedianAge)))
                .Append(mLContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(InputCaliforniaHosungPrice.TotalRooms)))
                .Append(mLContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(InputCaliforniaHosungPrice.TotalBedRooms)))
                .Append(mLContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(InputCaliforniaHosungPrice.Population)))
                .Append(mLContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(InputCaliforniaHosungPrice.HouseHolds)))
                .Append(mLContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(InputCaliforniaHosungPrice.MedianIncome)))
                .Append(mLContext.Transforms.Concatenate(
                    "Features",
                    "LatitudeEncoded",
                    "LongitudeEncoded",
                    nameof(InputCaliforniaHosungPrice.HousingMedianAge),
                    nameof(InputCaliforniaHosungPrice.TotalRooms),
                    nameof(InputCaliforniaHosungPrice.TotalBedRooms),
                    nameof(InputCaliforniaHosungPrice.Population),
                    nameof(InputCaliforniaHosungPrice.HouseHolds),
                    nameof(InputCaliforniaHosungPrice.MedianIncome),
                    "OceanProximityEncoded"
                    ));
        }

        public override IDataView LoadTestDataView()
        {
            DatabaseLoader loader = mLContext.Data.CreateDatabaseLoader<InputCaliforniaHosungPrice>();
            string connectionString = "Data Source=.;Initial Catalog=MachineLearning;Integrated Security=True;MultipleActiveResultSets=True";
            string sqlCommand = "select top 50 percent CAST(longitude as REAL) as Longitude, CAST(latitude as REAL) as Latitude, CAST(housing_median_age as REAL) as HousingMedianAge, CAST(total_rooms as REAL) as TotalRooms, CAST(total_bedrooms as REAL) as TotalBedRooms, CAST(population as REAL) as Population, CAST(households as REAL) as HouseHolds, CAST(median_income as REAL) as MedianIncome, CAST(median_house_value as REAL) as MedianHouseValue, ocean_proximity as OceanProximity from california_housing_price order by id desc";
            DatabaseSource databaseSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, sqlCommand);
            return loader.Load(databaseSource);

            //return mLContext.Data.LoadFromTextFile<InputCaliforniaHosungPrice>(@"C:\MLData\california-housing-price-test.csv", hasHeader: true, separatorChar: ',');
        }

        public override IDataView LoadTrainDataView()
        {
            DatabaseLoader loader = mLContext.Data.CreateDatabaseLoader<InputCaliforniaHosungPrice>();
            string connectionString = "Data Source=.;Initial Catalog=MachineLearning;Integrated Security=True;MultipleActiveResultSets=True";
            string sqlCommand = "select top 50 percent CAST(longitude as REAL) as Longitude, CAST(latitude as REAL) as Latitude, CAST(housing_median_age as REAL) as HousingMedianAge, CAST(total_rooms as REAL) as TotalRooms, CAST(total_bedrooms as REAL) as TotalBedRooms, CAST(population as REAL) as Population, CAST(households as REAL) as HouseHolds, CAST(median_income as REAL) as MedianIncome, CAST(median_house_value as REAL) as MedianHouseValue, ocean_proximity as OceanProximity from california_housing_price";
            DatabaseSource databaseSource = new DatabaseSource(SqlClientFactory.Instance, connectionString, sqlCommand);
            return loader.Load(databaseSource);

            //return mLContext.Data.LoadFromTextFile<InputCaliforniaHosungPrice>(@"C:\MLData\california-housing-price-train.csv", hasHeader: true, separatorChar: ',');
        }

        public override EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> SelectMLAlgo(EstimatorChain<ColumnConcatenatingTransformer> estimatorChain)
        {
            return estimatorChain.Append(mLContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"));
        }
    }
}
