import java.awt.EventQueue;
import visualisation.PointsEx;

/**
 * Main class to execute examples or the visualisation.
 * 
 * @author Vincent
 *
 */
public class Main {

	/**
	 * Main method. Currently runs visualization. 
	 * 
	 * @param args
	 *            Command-line arguments
	 */
	public static void main(String[] args) {
		// Visualisation

		EventQueue.invokeLater(new Runnable() {
			@Override
			public void run() {

				PointsEx ex = new PointsEx();
				ex.setVisible(true);
			}
		});
	}
}
