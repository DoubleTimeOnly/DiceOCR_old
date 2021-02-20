using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Runtime.InteropServices;

namespace DiceOCR_App
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        [DllImport(@"C:\Users\Victor2\Dropbox\DiceOCR\x64\Debug\DiceOCR.dll", EntryPoint = "mixed_mode_multiply", CallingConvention = CallingConvention.StdCall)]
        public static extern int Multiply(int x, int y);
        private void Button_Click(object sender, RoutedEventArgs e)
        {
            int result = Multiply(7, 7);
            Console.WriteLine("The answer is {0}", result);
        }
    }
}
