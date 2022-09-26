using MachineLearning.Shared.Common;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning.CaliforniaHosingPrice.Predictor
{
    public class MLModelPredictor : BaseMLModelPredictor
    {
        public MLModelPredictor(MLContext mLContext) : base(mLContext, @"C:\MLData\CaliforniaHousingPriceModel.zip")
        {
        }

        public override dynamic GeneratePredictionEngine(ITransformer trainedModel)
        {
            return mLContext.Model.CreatePredictionEngine<InputCaliforniaHosungPrice, OutputCaliforniaHosungPrice>(trainedModel);
        }

        public override ITransformer LoadTrainedModel()
        {
            return mLContext.Model.Load(fileName, out var modelInputSchema);
        }

        public override void Prediction(dynamic predictionEngine)
        {
            OutputCaliforniaHosungPrice result = predictionEngine.Predict(new InputCaliforniaHosungPrice()
            {
                Longitude = -121.24f,
                Latitude = 33.84f,
                HousingMedianAge = 9.0f,
                TotalRooms = 10484.0f,
                TotalBedRooms = 1603.0f,
                Population = 4005.0f,
                HouseHolds = 1419.0f,
                MedianIncome = 8.3931f,
                OceanProximity = "<1H OCEAN",
                MedianHouseValue = 0f, // To Predict. Actual/Obsered = 365300.0
            });

            Console.WriteLine($"Predicted Fare: {result.MedianHouseValue}, actual Fare: 365300.0");
        }
    }
}
