using Microsoft.Extensions.DependencyInjection;
using System;

namespace MachineLearning.CaliforniaHosingPrice.Predictor
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
