package main_package;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.event.DocumentEvent;

public class SwingPaint {
	JButton clearBtn, blackBtn, doneBtn, newNumberBtn;
	JTextField fileNameTextBox, offSetTextBox;
	String fileName;
	DrawArea drawArea;
	int c, offset;
	ActionListener actionListener = new ActionListener() {
		public void actionPerformed(ActionEvent e) {
			if (e.getSource() == clearBtn) {
				drawArea.clear();
			}else if (e.getSource() == blackBtn) {
				drawArea.black();
			}else if (e.getSource() == newNumberBtn){
				c = 0;
			}else if (e.getSource() == doneBtn) {
				System.out.println("Saving picture");
				
				Path path = Paths.get("./saved_img/");
				try {
					Files.createDirectories(path);
				} catch (IOException e1) {
					e1.printStackTrace();
				}
				if (offSetTextBox.getText() != "") {
					offset = Integer.parseInt(offSetTextBox.getText());
				}
				
				String sPath = "./saved_img/";
				int numberImg = c + offset;
				fileName = fileNameTextBox.getText() + "_" + numberImg;
				String extension = ".png";
				c = c + 1;
				drawArea.savePic(drawArea.getImage(), "png", (sPath + fileName + extension));
				System.out.println("filename = " + fileName + extension);
				drawArea.clear();
			}
		}
	};
	
	
	
	public static void main(String[] args) {
		
		new SwingPaint().show();
			
	}
	
	public void show() {
		//create main frame
		JFrame frame = new JFrame("Swing Paint");
		Container content = frame.getContentPane();
		//set layout on content pane
		content.setLayout(new BorderLayout());
		//create draw area
		drawArea = new DrawArea();
		
		//add to content pane
		content.add(drawArea, BorderLayout.CENTER);
		
		//create controls to apply colors and call clear feature
		
		JPanel controls = new JPanel();
		clearBtn = new JButton("Clear");
		clearBtn.addActionListener(actionListener);
		blackBtn = new JButton("Black");
		blackBtn.addActionListener(actionListener);
		doneBtn = new JButton("Save");
		doneBtn.addActionListener(actionListener);
		newNumberBtn = new JButton("New number");
		newNumberBtn.addActionListener(actionListener);
		
		//create text field for the file name
		JLabel fileNameLabel = new JLabel("Number :");
		fileNameTextBox = new JTextField("", 5);
		
		JLabel offSetLabel = new JLabel("Offset :");
		offSetTextBox = new JTextField("", 5);

		
		
		//add to panel
		controls.add(clearBtn);
		controls.add(blackBtn);
		controls.add(doneBtn);
		controls.add(newNumberBtn);
		
		controls.add(fileNameLabel);
		controls.add(fileNameTextBox);
		
		controls.add(offSetLabel);
		controls.add(offSetTextBox);
		
		//add to content pane
		content.add(controls, BorderLayout.NORTH);
		frame.setSize(600,600);
		//can close frame
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		// show the swing paint result
		frame.setVisible(true);
	}
}