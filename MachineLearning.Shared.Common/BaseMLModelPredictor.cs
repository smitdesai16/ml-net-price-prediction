using Microsoft.ML;

namespace MachineLearning.Shared.Common
{
    public abstract class BaseMLModelPredictor
    {
        public MLContext mLContext;
        public string fileName;
        public BaseMLModelPredictor(MLContext mLContext, string fileName)
        {
            this.mLContext = mLContext;
            this.fileName = fileName;
        }

        public void Predict()
        {
            ITransformer trainedModel = LoadTrainedModel();
            var predictionEngine = GeneratePredictionEngine(trainedModel);
            Prediction(predictionEngine);
        }

        public abstract ITransformer LoadTrainedModel();
        public abstract dynamic GeneratePredictionEngine(ITransformer trainedModel);
        public abstract void Prediction(dynamic predictionEngine);
    }
}
