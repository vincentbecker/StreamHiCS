package visualisation;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Random;

import javax.swing.JPanel;
import javax.swing.Timer;

import org.apache.commons.math3.util.MathArrays;

import fullsystem.Contrast;
import microclusters.Microcluster;
import streamdatastructures.MCAdapter;
import streamdatastructures.Selection;
import streamdatastructures.SummarisationAdapter;
import weka.core.DenseInstance;
import weka.core.Instance;

class CentroidSurface extends JPanel implements ActionListener {

	/**
	 * Default serial version id.
	 */
	private static final long serialVersionUID = 1L;
	private final int DELAY = 10;
	private Timer timer;
	private Contrast contrast;
	private int count = 0;
	private int conceptChange = 1100;
	private double xRange = 10;
	private double yRange = 10;
	private Random r;
	private int[] shuffledDimensions = { 0, 1 };
	private SummarisationAdapter adapter;

	public CentroidSurface() {
		adapter = new MCAdapter(1000, 0.2, 0.2, "adapting");
		this.contrast = new Contrast(20, 0.4, adapter);
		r = new Random();
		initTimer();
	}

	private void initTimer() {

		timer = new Timer(DELAY, this);
		timer.start();
	}

	public Timer getTimer() {
		return timer;
	}

	private void doDrawing(Graphics g) {
		// Add a new instance
		createAndAddInstance();

		Graphics2D g2d = (Graphics2D) g;

		g2d.setPaint(Color.blue);

		int w = getWidth();
		int h = getHeight();

		// Draw each centroid
		double[] vector;
		int xPixel = 0;
		int yPixel = 0;
		int weight = 0;
		Microcluster[] cs = ((MCAdapter) adapter).getCentroids();
		Microcluster c;
		boolean drawSlice = (count % 100 == 0);
		Selection s = null;
		if (drawSlice) {
			// Shuffle dimensions
			MathArrays.shuffle(shuffledDimensions);
			// System.out.println("[" + shuffledDimensions[0] + ", " +
			// shuffledDimensions[1] + "]");
			// System.out.println("Number of centroids: " + cs.length);
			s = adapter.getSliceIndexes(shuffledDimensions, 0.2);
			// System.out.println("Selected Indexes: " + s.toString());
			// System.out.println("Selected IDs: ");
			// for(int i = 0; i < s.size(); i++){
			// System.out.print(cs[i].getId() + ", ");
			// }
		}
		for (int i = 0; i < cs.length; i++) {
			c = cs[i];
			if (drawSlice && s.contains(i)) {
				g2d.setColor(Color.RED);
			}
			vector = c.getCentre();
			xPixel = (int) (w * (vector[0] / xRange));
			yPixel = (int) (h * (vector[1] / yRange));
			weight = (int) (c.getWeight(count) * 10);
			g2d.drawOval(xPixel, yPixel, weight, weight);
			// g2d.drawString(c.getId() + "", xPixel, yPixel);
			if (drawSlice) {
				g2d.setColor(Color.BLUE);
			}
			// System.out.println("x: " + xPixel + " y: " + yPixel);
		}
	}

	@Override
	public void paintComponent(Graphics g) {

		super.paintComponent(g);
		doDrawing(g);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		revalidate();
		repaint();
	}

	private void createAndAddInstance() {
		double x = r.nextGaussian();
		double y = r.nextGaussian();
		double offset = 0;
		if (count >= conceptChange) {
			offset = 10;
		} else {
			if (count % 2 == 0) {
				offset = 3;
			} else {
				offset = 7;
			}
		}

		x += offset;
		y += offset;
		addInstance(x, y);
		count++;
	}

	private void addInstance(double x, double y) {
		Instance inst = new DenseInstance(2);
		inst.setValue(0, x);
		inst.setValue(1, y);
		contrast.add(inst);
	}
}
