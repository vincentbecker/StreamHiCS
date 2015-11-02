package visualisation;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.Locale;

import javax.swing.JFrame;
import javax.swing.Timer;

public class PointsEx extends JFrame {

	/**
	 * Default serial version id.
	 */
	private static final long serialVersionUID = 1L;

	public PointsEx() {

		initUI();
	}

	private void initUI() {

		Locale.setDefault(Locale.ENGLISH);
		//final CentroidSurface surface = new CentroidSurface();
		final MicroclusterSurface surface = new MicroclusterSurface();
		add(surface);

		addWindowListener(new WindowAdapter() {
			@Override
			public void windowClosing(WindowEvent e) {
				Timer timer = surface.getTimer();
				timer.stop();
			}
		});

		setTitle("Micro-Clusters");
		setSize(700, 700);
		setLocationRelativeTo(null);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	}
}