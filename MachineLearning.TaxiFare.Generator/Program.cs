using Microsoft.Extensions.DependencyInjection;

namespace MachineLearning.TaxiFare.Generator
{
    internal class Program
    {
        static void Main()
        {
            var serviceProvider = AutofacContainerConfiguration.Configure();
            serviceProvider.GetService<MLModelGenerator>().Generate();
        }
    }
}
