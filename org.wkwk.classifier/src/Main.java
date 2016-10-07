
import java.util.Scanner;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Husni
 */
public class Main {
    public static void main(String[] args) throws Exception {
        MenuHandler menuHandler = new MenuHandler();
        Scanner scanner = new Scanner(System.in);
        int menu = 0;
        while (menu != 8) {
            System.out.println("============================");
            System.out.println("=====  WKWK Classifier =====");
            System.out.println("============================");
            System.out.println("1. Load arff data");
            System.out.println("2. Remove attribute");
            System.out.println("3. Resample");
            System.out.println("4. Build classifier");
            System.out.println("5. Test model");
            System.out.println("6. Save model");
            System.out.println("7. Load model");
            System.out.println("8. Exit");
            
            menu = scanner.nextInt();
            try {
                switch (menu) {
                    case 1:
                        menuHandler.loadDataMenu();
                        break;
                    case 2:
                        menuHandler.resampleMenu();
                    case 3:
                        menuHandler.removeAttributeMenu();
                        break;
                    case 4:
                        menuHandler.buildClassifierMenu();
                        break;
                    case 5:
                        menuHandler.testModelMenu();
                        break;
                    case 6:
                        menuHandler.saveModelMenu();
                        break;
                    case 7:
                        menuHandler.loadModelMenu();
                        break;
                }
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        }
    }
}
