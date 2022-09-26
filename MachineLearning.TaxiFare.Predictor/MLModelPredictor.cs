using MachineLearning.Shared.Common;
using Microsoft.ML;
using System;

namespace MachineLearning.TaxiFare.Predictor
{
    public class MLModelPredictor : BaseMLModelPredictor
    {
        public MLModelPredictor(MLContext mLContext) : base(mLContext, @"C:\MLData\TaxiFareModel.zip")
        {
        }

        public override dynamic GeneratePredictionEngine(ITransformer trainedModel)
        {
            return mLContext.Model.CreatePredictionEngine<InputTaxiFare, OutputTaxiFare>(trainedModel);
        }

        public override ITransformer LoadTrainedModel()
        {
            return mLContext.Model.Load(fileName, out var modelInputSchema);
        }

        public override void Prediction(dynamic predictionEngine)
        {
            OutputTaxiFare result = predictionEngine.Predict(new InputTaxiFare()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            });

            Console.WriteLine($"Predicted Fare: {result.FareAmount}, actual Fare: 15.5");
        }
    }
}
