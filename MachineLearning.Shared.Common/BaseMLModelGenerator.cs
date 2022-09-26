using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning.Shared.Common
{
    public abstract class BaseMLModelGenerator
    {
        public MLContext mLContext;
        public string fileName;
        public BaseMLModelGenerator(MLContext mLContext, string fileName)
        {
            this.mLContext = mLContext;
            this.fileName = fileName;
        }

        public void Generate()
        {
            IDataView trainDataView = LoadTrainDataView();
            IDataView testDataView = LoadTestDataView();
            IDataView cleanTrainDataView = CleanTrainDataView(trainDataView);
            EstimatorChain<ColumnConcatenatingTransformer> dataProcessPipeline = GetDataProcessPipeline();
            EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> trainingPipeline = SelectMLAlgo(dataProcessPipeline);
            var trainedModel = Train(trainingPipeline, cleanTrainDataView);
            RegressionMetrics metric = Evaluate(trainedModel, testDataView);
            Console.WriteLine(metric.RSquared);
            Save(trainedModel, cleanTrainDataView.Schema, fileName);
        }

        public abstract IDataView LoadTestDataView();
        public abstract IDataView LoadTrainDataView();
        public abstract IDataView CleanTrainDataView(IDataView trainDataView);
        public abstract EstimatorChain<ColumnConcatenatingTransformer> GetDataProcessPipeline();
        public abstract EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> SelectMLAlgo(EstimatorChain<ColumnConcatenatingTransformer> estimatorChain);

        public dynamic Train(EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> trainingPipeline, IDataView cleanTrainDataView)
        {
            return trainingPipeline.Fit(cleanTrainDataView);
        }

        public RegressionMetrics Evaluate(dynamic trainedModel, IDataView testDataView)
        {
            IDataView predictions = trainedModel.Transform(testDataView);
            return mLContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
        }

        public void Save(dynamic trainedModel, DataViewSchema dataViewSchema, string filename)
        {
            mLContext.Model.Save(trainedModel, dataViewSchema, filename);
        }
    }
}
