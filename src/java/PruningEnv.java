// Environment code for project pruning

import jason.asSyntax.*;
import jason.environment.*;
import jason.jeditplugin.AgentSpeakSideKickParser;

import java.util.logging.*;
import java.util.stream.Collectors;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

public class PruningEnv extends Environment {

    private Logger logger = Logger.getLogger("pruning."+PruningEnv.class.getName());
    
    private Literal remainingL = Literal.parseLiteral("remainingLayers");
    
    private Literal undoPruning = Literal.parseLiteral("decreasePerformance");
    
    private Literal trainAgain = Literal.parseLiteral("trainAgain");
    
    
    private Model model;
    private Process process;
    private int wrongPruning = 0;
    private double lastTrainedResult = 0.0;
    private double lastPrunedResult = 0.0;
    
    private double tau1 = 0.2;
    private double tau2 = 0.07;

    /** Called before the MAS execution with the args informed in .mas2j */
    @Override
    public void init(String[] args) {
        super.init(args);
        addPercept(this.remainingL);
        
        // Training the neural network by first time
        runCommand("python environment\\train.py --first_time");
        
        // Loading the CNN configuration and instantiating Model
        CSV m = new CSV("wrapping\\model.csv");
        this.model = new Model(m);
        this.lastTrainedResult = readPerformance("wrapping\\result.txt");
	this.lastPrunedResult = this.lastTrainedResult
        
        CSV pruned = new CSV("wrapping\\pruned_layers.csv");
    }

    @Override
    public boolean executeAction(String agName, Structure action) {
    	System.out.println("\tAgent "+agName+" is doing "+action);
        
        /* Performing the actions */
        if (action.getFunctor().equals("make_prune")) {
        	prune();
        	
        } else if (action.getFunctor().equals("verify")) {
        	clearAllPercepts();

        	double perf = evaluatePerformance();
        	
        	 if (this.wrongPruning >= 10 || this.model.size() == 0) { // After, just_end
         		System.out.println("\tverify_case: 4");
         		
         	} else if ( (this.lastPrunedResult - perf) <= (this.tau2 * this.lastPrunedResult) ) { // After, continue_pruning
        		System.out.println("\tverify_case: 1");
        		this.lastPrunedResult = perf;
        		addPercept(this.remainingL);
        	
        	} else if (perf < this.tau1 * this.lastTrainedResult) { // after, train
        		System.out.println("\tcounter: verify_case: 3");
        		addPercept(this.remainingL);
        		addPercept(this.undoPruning);
        		addPercept(this.trainAgain);
        		
        	} else { // After, undo_prune
        		System.out.println("\tcounter: verify_case: 2");
        		addPercept(this.remainingL);
        		addPercept(this.undoPruning);
        	}
        	
        } else if (action.getFunctor().equals("train")) {
        	runCommand("python environment\\train.py");
        	this.lastTrainedResult = readPerformance("wrapping\\result.txt");
		this.lastPrunedResult = this.lastTrainedResult
        	this.wrongPruning = 0;
        	
        } else if (action.getFunctor().equals("undo_prune")) {
    		undoPrune();
    		
        } else if (action.getFunctor().equals("just_end")) {
        	System.out.println("\tProcesso finalizado");
        	try {
    			TimeUnit.SECONDS.sleep(8);
    		} catch (Exception e) {
    			e.printStackTrace();
    		}
        	System.exit(0);
        }
        
        informAgsEnvironmentChanged("bob"); 
        return true;
    }

    /** Called before the end of MAS execution */
    @Override
    public void stop() {
        super.stop();
    }
    
    public double readPerformance(String path) {
    	String performance = "";
    	double result;
    	
    	try(BufferedReader br = new BufferedReader(new FileReader(path))) {
    	    StringBuilder sb = new StringBuilder();
    	    String line = br.readLine();

    	    while (line != null) {
    	        sb.append(line);
    	        sb.append(System.lineSeparator());
    	        line = br.readLine();
    	    }
    	    performance = sb.toString();
    	} catch (IOException e) {
    		e.printStackTrace();
    	}
    	
    	result = Double.valueOf(performance);
    	
    	return result;
    }
    
    public void runCommand(String command) {
    	
    	try {
    		// Using runtime to run python
    		this.process = Runtime.getRuntime().exec(command);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
	    
    }
        
    public void undoPrune() {
    	CSV pruned = new CSV("wrapping\\pruned_layers.csv");
    	
    	// Removing from the list
    	String[] undo = pruned.getLast();
    	pruned.removeLast();
    	System.out.println("\t\t\tRemoving element: " + undo[0] + " - " + undo[1] + " - " + undo[2]);
    	pruned.save();
    	
    	// Re-inserting the layer in Model
    	int layer = Integer.valueOf(undo[0]);
    	int channel = Integer.valueOf(undo[1]);
    	//this.model.getLayer(layer).addChannel(channel);
	runCommand("python environment\\update_mask.py");
    }
    
    public void prune() {
    	String[] result = this.model.choose_channel();
    	String[] pruned_channel = new String[3];
    	pruned_channel[0] = result[0]; pruned_channel[1] = result[1]; pruned_channel[2] = "0";
    	if (pruned_channel[1].equals("-1")) {
    		this.wrongPruning++;
    	}
    	else {
    		this.wrongPruning = 0;
    		CSV pruned = new CSV("wrapping\\pruned_layers.csv");
    		pruned.add(pruned_channel);
        	pruned.save();
    	}
	runCommand("python environment\\update_mask.py");
    }
    
    public double evaluatePerformance() {
    	runCommand("python environment\\validate.py");
    	CSV pruned = new CSV("wrapping\\pruned_layers.csv");
    	return Double.parseDouble(pruned.getLast()[2]);
    }
    
    
    /**
     * Implementation of Model. Controls the which layers/channels will be choosed to prune
     * @author Andrey de Aguiar Salvi
     *
     */
    class Model {
    	ArrayList<Layer> layers = new ArrayList();
    	
    	public Model (CSV model) { // Each line from CSV is one layers, containing the number of channels
    		int size = model.size();
    		for (int i = 0; i < size; i++) {
    			layers.add(new Layer( Integer.valueOf(model.get(i, 0)) ));
    		}
    	}
    	
    	public Layer getLayer(int i ) {
    		return layers.get(i);
    	}
    	
    	public int size() {
    		int size = 0;
    		for (Layer l : this.layers) size += l.size();
    		return size;
    	}
    	
    	public String[] choose_channel() {
    		String[] result = new String[2];
    		double prob = ThreadLocalRandom.current().nextDouble();
    		int channel;
    		if(prob <= 0.4) {
    			channel = this.layers.get(0).choose_channel();
    			result[0] = "0";
    			result[1] = String.valueOf(channel);
    		} else if (prob <= 0.7) {
    			channel = this.layers.get(1).choose_channel();
    			result[0] = "1";
    			result[1] = String.valueOf(channel);
    		} else if (prob <= 0.9) {
    			channel = this.layers.get(2).choose_channel();
    			result[0] = "2";
    			result[1] = String.valueOf(channel);
    		} else {
    			channel = this.layers.get(3).choose_channel();
    			result[0] = "3";
    			result[1] = String.valueOf(channel);
    		}
    		return result;
    	}
    }
    
    /**
     * Implementation of Layers. Encapsulate channels
     * @author Andrey de Aguiar Salvi
     *
     */
    class Layer {
    	ArrayList<Channel> channels = new ArrayList();
    	private int maxChooses = 5;
    	
    	public Layer (int nChannels) {
    		for (int i = 0; i < nChannels; i++) {
    			Channel ch = new Channel();
    			this.channels.add(ch);
    		}
    	}
    	
    	public Channel getChannel(int i) {
    		return this.channels.get(i);
    	}
    	
    	public int size() {
    		return this.channels.size();
    	}
    	
    	public int choose_channel() {
    		if (size() == 0) return -1;
    		int channel = ThreadLocalRandom.current().nextInt(0, size());
    		this.channels.get(channel).choose();
    		if (this.channels.get(channel).timesChoosen == this.maxChooses) this.channels.remove(channel);
    		return channel;
    	}
    	
    	public void addChannel(int index) {
    		Channel ch = new Channel();
    		this.channels.add(index, ch);
    	}
    }
    
    /**
     * Implementation of Channel. Currently saves how many times it was chosen.
     * @author Andrey de Aguiar Salvi
     *
     */
    class Channel {
    	int timesChoosen = 0;
    	
    	public void choose() {
    		this.timesChoosen++;
    	}
    }
    
    /**
     * Class CSV, trying to mitigate the lack of Pandas in Java
     * @author Andrey de Aguiar Salvi
     *
     */
    class CSV {

    	private ArrayList<String[]> matrix;
    	private String path;
    	
    	public CSV(String path) {
    		this.path = path;
    		try {
    			List<String[]> rowList = new ArrayList<String[]>();
    			BufferedReader br = new BufferedReader(new FileReader(path));
    		    String line;
    		    while ((line = br.readLine()) != null) {
    		        String[] lineItems = line.split(",");
    		        rowList.add(lineItems);
    		    }
    		    br.close();
    		    this.matrix = new ArrayList<String[]>();
    		    for (int i = 0; i < rowList.size(); i++) {
    		    	String[] row = rowList.get(i);
    		    	this.matrix.add(row);
    		    }
    		}
    		catch(Exception e){
    		    e.printStackTrace();
    		}
    	}
    	
    	public int size() {
    		return this.matrix.size();
    	}
    	
    	public String get(int row, int col) {
    		return this.matrix.get(row)[col];
    	}
    	
    	public String[] get(int row) {
    		return this.matrix.get(row);
    	}
    	
    	public String[] getLast() {
    		return this.matrix.get(this.matrix.size() -1);
    	}
    	
    	public void add(String[] row) {
    		this.matrix.add(row);
    	}
    	
    	public void remove(int row) {
    		this.matrix.remove(row);
    	}
    	
    	public void removeLast() {
    		int row = this.matrix.size() -1;
    		this.matrix.remove(row);
    	}
    	
    	public void save() {
    		try {
				FileWriter writer = new FileWriter(this.path, false);
				for (String[] row : this.matrix) {
					writer.write(Arrays.asList(row).stream().collect(Collectors.joining(",")));
		            writer.write("\n"); // newline
				}
				writer.close();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
    	}
    }
}

