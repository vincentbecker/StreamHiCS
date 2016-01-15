package environment;

import java.util.HashMap;

public class Stopwatch {

	HashMap<String, NamedWatch> watches = new HashMap<String, NamedWatch>();

	public void start(String name) {
		if (!watches.containsKey(name)) {
			watches.put(name, new NamedWatch(name));
		}
		watches.get(name).start();
	}

	public void stop(String name) {
		if (watches.containsKey(name)) {
			watches.get(name).stop();
		}
	}

	public double getTime(String name) {
		if (watches.containsKey(name)) {
			return watches.get(name).getTotalTime();
		}
		return -1;
	}

	public void reset() {
		for (NamedWatch w : watches.values()) {
			w.reset();
		}
	}

	@Override
	public String toString() {
		String res = "";

		for (NamedWatch w : watches.values()) {
			res += w.toString() + "; ";
		}

		return res;
	}

	private static class NamedWatch {
		private String name;
		private long totalTime;
		private long beginning;
		private long end;
		private boolean running = false;

		private NamedWatch(String name) {
			this.name = name;
		}

		private double getTotalTime() {
			return ((double) totalTime) / 1000000000;
		}

		private void start() {
			beginning = System.nanoTime();
			running = true;
		}

		private void stop() {
			if (running) {
				end = System.nanoTime();
				totalTime += (end - beginning);
				running = false;
			}
		}

		private void reset() {
			running = false;
			beginning = 0;
			end = 0;
			totalTime = 0;
		}

		@Override
		public String toString() {
			return name + ": " + getTotalTime() + "s";
		}
	}
}
