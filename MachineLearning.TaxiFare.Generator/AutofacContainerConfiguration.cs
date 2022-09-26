using Autofac;
using Autofac.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning.TaxiFare.Generator
{
    public class AutofacContainerConfiguration
    {
        public static IServiceProvider Configure()
        {
            var serviceCollection = new ServiceCollection();
            var containerBuilder = new ContainerBuilder();
            containerBuilder.Populate(serviceCollection);
            containerBuilder.RegisterType<MLModelGenerator>().SingleInstance();
            containerBuilder.RegisterType<MLContext>().SingleInstance();

            var container = containerBuilder.Build();
            return new AutofacServiceProvider(container);
        }
    }
}
