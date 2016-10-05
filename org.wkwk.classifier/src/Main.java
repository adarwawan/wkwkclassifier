
import java.util.Scanner;

/**
 *
 * @author Husni
 */
public class Main {
    public static void main(String[] args) {
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
            switch(menu) {
                case 1:
                    System.out.println("Load arff data menu");
                    break;
                case 2:
                    System.out.println("remove attribute menu");
                    break;
                case 3:
                    System.out.println("build classifier menu");
                    break;
                case 4:
                    System.out.println("test model menu");
                    break;
                case 5:
                    System.out.println("save model menu");
                    break;
                case 6: 
                    System.out.println("load model menu");
                    break;
            }
        }
    }
}
