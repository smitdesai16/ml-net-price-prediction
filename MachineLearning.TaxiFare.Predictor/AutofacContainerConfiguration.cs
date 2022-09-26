using Autofac.Extensions.DependencyInjection;
using Autofac;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning.TaxiFare.Predictor
{
    public class AutofacContainerConfiguration
    {
        public static IServiceProvider Configure()
        {
            var serviceCollection = new ServiceCollection();
            var containerBuilder = new ContainerBuilder();
            containerBuilder.Populate(serviceCollection);
            containerBuilder.RegisterType<MLModelPredictor>().SingleInstance();
            containerBuilder.RegisterType<MLContext>().SingleInstance();

            var container = containerBuilder.Build();
            return new AutofacServiceProvider(container);
        }
    }
}
