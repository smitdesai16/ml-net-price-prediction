using Microsoft.Extensions.DependencyInjection;
using System;

namespace MachineLearning.TaxiFare.Predictor
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var serviceProvider = AutofacContainerConfiguration.Configure();
            serviceProvider.GetService<MLModelPredictor>().Predict();
        }
    }
}
