package drawing;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Random;

import javax.swing.JPanel;
import javax.swing.Timer;

import org.apache.commons.math3.util.MathArrays;

import centroids.Centroid;
import contrast.Callback;
import contrast.CentroidContrast;
import streamDataStructures.Selection;
import weka.core.DenseInstance;
import weka.core.Instance;

class Surface extends JPanel implements ActionListener {

	private final int DELAY = 100;
	private Timer timer;
	private CentroidContrast centroidContrast;
	private Callback callback = new Callback() {

		@Override
		public void onAlarm() {
			System.out.println("Alarm.");
		}

	};
	private int count = 0;
	private double xRange = 10;
	private double yRange = 10;
	private Random r;
	private int[] shuffledDimensions = { 0, 1 };

	public Surface() {
		this.centroidContrast = new CentroidContrast(callback, 2, 20, 0.4, 10000);
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
		Centroid[] cs = centroidContrast.getCentroids();
		Centroid c;
		boolean drawSlice = (count % 100 == 0);
		Selection s = null;
		if (drawSlice) {
			// Shuffle dimensions
			MathArrays.shuffle(shuffledDimensions);
			// System.out.println("[" + shuffledDimensions[0] + ", " +
			// shuffledDimensions[1] + "]");
			// System.out.println("Number of centroids: " + cs.length);
			s = centroidContrast.getSliceIndexes(shuffledDimensions, 0.4);
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
			vector = c.getVector();
			xPixel = (int) (w * (vector[0] / xRange));
			yPixel = (int) (h * (vector[1] / yRange));
			weight = (int) (c.getWeight() * 10);
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
		if (count % 2 == 0) {
			offset = 3;
		} else {
			offset = 7;
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
		centroidContrast.add(inst);
	}
}
