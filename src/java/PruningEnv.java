// Environment code for project pruning

import jason.asSyntax.*;
import jason.environment.*;
import jason.jeditplugin.AgentSpeakSideKickParser;

import java.util.logging.*;
import java.util.stream.Collectors;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
public class PruningEnv extends Environment {

    private Logger logger = Logger.getLogger("pruning."+PruningEnv.class.getName());
    
    private Literal remainL = Literal.parseLiteral("remaingLayers");
    
    private Literal undoPruning = Literal.parseLiteral("decreasePerformance");
    
    private Literal trainAgain = Literal.parseLiteral("trainAgain");
    
    private Model M;
    private int counter = 0;
    private boolean continue_pruning = true;
    private boolean stop = false;
    private Process process;

    /** Called before the MAS execution with the args informed in .mas2j */
    @Override
    public void init(String[] args) {
        super.init(args);
        addPercept(this.remainL);
        this.M = new Model();
    }

    @Override
    public boolean executeAction(String agName, Structure action) {
    	System.out.println("\tAgent "+agName+" is doing "+action);
        
        /* Performing the actions */
        if (action.getFunctor().equals("make_prune")) {
        	prune();
        	
        } else if (action.getFunctor().equals("verify")) {
        	clearAllPercepts();

        	this.continue_pruning = evaluatePerformance();
        	
        	if (this.continue_pruning && this.counter <= 12) { // After, continue_pruning
        		System.out.println("\tcounter: " + this.counter + " verify_case: 1");
        		addPercept(this.remainL);
        	} else if (!this.continue_pruning && this.counter <= 12) { // After, undo_prune
        		System.out.println("\tcounter: " + this.counter + " verify_case: 2");
        		addPercept(this.remainL);
        		addPercept(this.undoPruning);
        	} else if (this.continue_pruning && this.counter > 12) { // after, train
        		System.out.println("\tcounter: " + this.counter + " verify_case: 3");
        		addPercept(this.remainL);
        		addPercept(this.undoPruning);
        		addPercept(this.trainAgain);
        	} else if (!this.continue_pruning && this.counter > 12) { // After, just_end
        		System.out.println("\tcounter: " + this.counter + " verify_case: 4");
        		this.stop = true;
        	}
        	
        } else if (action.getFunctor().equals("train")) {
//        	runCommand("cmd cd environment && python test.py");
//        	runCommand("cmd python -m environment\test.py");
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
    
    public void runCommand(String command) {
    	
    	try {
    		// Trying with Runtime
    		this.process = Runtime.getRuntime().exec(command);
//            this.process.waitFor();
            
        	// Trying with ProcessBuilder
//            ProcessBuilder pb = new ProcessBuilder("cmd", "cd environment", "python test.py");
//       	 	this.process = pb.start();
       	 
            BufferedReader reader = new BufferedReader(
            		new InputStreamReader(this.process.getInputStream())
            		); 
            String line; 
            while((line = reader.readLine()) != null) { 
                System.out.println(line);
            } 
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
//        } catch (InterruptedException e) {
//            // TODO Auto-generated catch block
//            e.printStackTrace();
        }
    
    }
    
    /**
     * Funções reais
     */
//    public boolean prune() {
//    	int i = ThreadLocalRandom.current().nextInt(0, this.M.size());
//    	Layer l = this.M.getLayer(i);
//    	
//    	int j = ThreadLocalRandom.current().nextInt(0, l.size());
//    	Item channel = l.getChannel(j);
//    	
//    	channel.nTimes += 1;
//    	channel.performance -= 5.0;
//    }
    
    public void undoPrune() {
    	CSV remaing = new CSV("wrapping\\remaing_layers.csv");
    	CSV pruned = new CSV("wrapping\\pruned_layers.csv");
    	
    	String[] undo = pruned.getLast();
    	pruned.removeLast();
    	System.out.println("\t\t\tRemoving element: " + undo[0] + " - " + undo[1] + " - " + undo[2]);
    	remaing.add(undo);
    	
    	remaing.save();
    	pruned.save();
    }
    
    
    /**
     * Funções mockadas
     */
    public void prune() {
    	this.counter += 1;
    }
    
    public boolean evaluatePerformance() {
    	if (this.counter % 3 == 0) return false;
    	return true;
    }
    
    
    /*
     * Implementation of a model
     */
    class Model {
    	ArrayList<Layer> layers = new ArrayList();
    	
    	public Model () {
    		for (int i = 0; i < 5; i++) {
    			layers.add(new Layer(15));
    		}
    	}
    	
    	public Layer getLayer(int i ) {
    		return layers.get(i);
    	}
    	
    	public int size() {
    		return layers.size();
    	}
    }
    
    /* 
     * Implementation of a Layer
     */
    class Layer {
    	ArrayList<Item> channels = new ArrayList();
    	
    	public Layer (int nChannels) {
    		for (int i = 0; i < nChannels; i++) {
    			Item it = new Item(i, 100.0);
    			this.channels.add(it);
    		}
    	}
    	
    	public Item getChannel(int i) {
    		return this.channels.get(i);
    	}
    	
    	public int size() {
    		return channels.size();
    	}
    }
    
    class Item {
    	public int index;
    	public double performance;
    	public int nTimes;
    	
		public Item(int index, double performance) {
    		this.index = index;
    		this.performance = performance;
    		this.nTimes = 0;
    	}
    }
    
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
    		    // Handle any I/O problems
    		}
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

