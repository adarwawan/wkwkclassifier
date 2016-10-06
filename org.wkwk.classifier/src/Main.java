
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
        while (menu != 7) {
            System.out.println("============================");
            System.out.println("=====  WKWK Classifier =====");
            System.out.println("============================");
            System.out.println("1. Load arff data");
            System.out.println("2. Remove attribute");
            System.out.println("3. Build classifier");
            System.out.println("4. Test model");
            System.out.println("5. Save model");
            System.out.println("6. Load model");
            System.out.println("7. Exit");
            
            menu = scanner.nextInt();
            try {
                switch (menu) {
                    case 1:
                        menuHandler.loadDataMenu();
                        break;
                    case 2:
                        menuHandler.removeAttributeMenu();
                        break;
                    case 3:
                        menuHandler.buildClassifierMenu();
                        break;
                    case 4:
                        menuHandler.testModelMenu();
                        break;
                    case 5:
                        menuHandler.saveModelMenu();
                        break;
                    case 6:
                        menuHandler.loadModelMenu();
                        break;
                }
            } catch (Exception e) {
                System.out.println(e.getMessage());
            }
        }
    }
}
