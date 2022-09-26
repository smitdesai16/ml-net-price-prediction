using Microsoft.Extensions.DependencyInjection;
using System;

namespace MachineLearning.CaliforniaHosingPrice.Generator
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var serviceProvider = AutofacContainerConfiguration.Configure();
            serviceProvider.GetService<MLModelGenerator>().Generate();
        }
    }
}
