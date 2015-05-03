using Microsoft.Owin;
using Owin;

[assembly: OwinStartupAttribute(typeof(TestWebApplication.Startup))]
namespace TestWebApplication
{
    public partial class Startup {
        public void Configuration(IAppBuilder app) {
            ConfigureAuth(app);
        }
    }
}
