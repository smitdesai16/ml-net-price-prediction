using MachineLearning.Shared.Common;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace MachineLearning.TaxiFare.Generator
{
    public class MLModelGenerator : BaseMLModelGenerator
    {
        public MLModelGenerator(MLContext mLContext) : base(mLContext, @"C:\MLData\TaxiFareModel.zip")
        {
        }

        public override IDataView CleanTrainDataView(IDataView trainDataView)
        {
            return mLContext.Data.FilterRowsByColumn(trainDataView, nameof(InputTaxiFare.FareAmount), lowerBound: 1, upperBound: 150);
        }

        public override EstimatorChain<ColumnConcatenatingTransformer> GetDataProcessPipeline()
        {
            return mLContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(InputTaxiFare.FareAmount))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: nameof(InputTaxiFare.VendorId)))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: nameof(InputTaxiFare.RateCode)))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: nameof(InputTaxiFare.PaymentType)))
                .Append(mLContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(InputTaxiFare.PassengerCount)))
                .Append(mLContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(InputTaxiFare.TripTime)))
                .Append(mLContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(InputTaxiFare.TripDistance)))
                .Append(mLContext.Transforms.Concatenate(
                    "Features",
                    "VendorIdEncoded",
                    "RateCodeEncoded",
                    "PaymentTypeEncoded",
                    nameof(InputTaxiFare.PassengerCount),
                    nameof(InputTaxiFare.TripTime),
                    nameof(InputTaxiFare.TripDistance)
                    ));
        }

        public override IDataView LoadTestDataView()
        {
            return mLContext.Data.LoadFromTextFile<InputTaxiFare>(@"C:\MLData\taxi-fare-test.csv", hasHeader: true, separatorChar: ',');
        }

        public override IDataView LoadTrainDataView()
        {
            return mLContext.Data.LoadFromTextFile<InputTaxiFare>(@"C:\MLData\taxi-fare-train.csv", hasHeader: true, separatorChar: ',');
        }

        public override EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> SelectMLAlgo(EstimatorChain<ColumnConcatenatingTransformer> estimatorChain)
        {
            return estimatorChain.Append(mLContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"));
        }
    }
}
