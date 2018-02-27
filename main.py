#!/usr/bin/env python3

import random
import time
import sys
import threading
from optparse import OptionParser

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtWidgets import QLabel, QTextEdit, QFrame
from PyQt5.QtWidgets import QPushButton, QSlider, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor

import gym
import gym_minigrid

from model.training import Model, Rollout, train_model, eval_model, run_model

class AIGameWindow(QMainWindow):
    """Application window for the baby AI game"""

    def __init__(self, env, model):
        super().__init__()
        self.initUI()

        # By default, manual stepping only
        self.fpsLimit = 0

        self.env = env
        self.lastObs = None

        self.model = model
        self.rollouts = []
        self.rollout = None

        self.stepTimer = QTimer()
        self.stepTimer.setInterval(0)
        self.stepTimer.setSingleShot(False)
        self.stepTimer.timeout.connect(self.stepClicked)

        self.resetEnv()

    def initUI(self):
        """Create and connect the UI elements"""

        self.resize(512, 512)
        self.setWindowTitle('Baby AI Game')

        # Full render view (large view)
        self.imgLabel = QLabel()
        self.imgLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        leftBox = QVBoxLayout()
        leftBox.addStretch(1)
        leftBox.addWidget(self.imgLabel)
        leftBox.addStretch(1)

        # Area on the right of the large view
        rightBox = self.createRightArea()

        # Arrange widgets horizontally
        hbox = QHBoxLayout()
        hbox.addLayout(leftBox)
        hbox.addLayout(rightBox)

        # Create a main widget for the window
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)
        mainWidget.setLayout(hbox)

        # Show the application window
        self.show()
        self.setFocus()

    def createRightArea(self):
        # Agent render view (partially observable)
        self.obsImgLabel = QLabel()
        self.obsImgLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        miniViewBox = QHBoxLayout()
        miniViewBox.addStretch(1)
        miniViewBox.addWidget(self.obsImgLabel)
        miniViewBox.addStretch(1)

        self.missionBox = QTextEdit()
        self.missionBox.setMinimumSize(500, 100)
        self.missionBox.textChanged.connect(self.missionEdit)

        buttonBox = self.createButtons()

        self.stepsLabel = QLabel()
        self.stepsLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.stepsLabel.setAlignment(Qt.AlignCenter)
        self.stepsLabel.setMinimumSize(60, 10)
        resetBtn = QPushButton("Reset")
        resetBtn.clicked.connect(self.resetBtn)
        trainBtn = QPushButton("Train")
        trainBtn.clicked.connect(self.train)
        stepsBox = QHBoxLayout()
        stepsBox.addStretch(1)
        stepsBox.addWidget(QLabel("Steps remaining"))
        stepsBox.addWidget(self.stepsLabel)
        stepsBox.addWidget(resetBtn)
        stepsBox.addWidget(trainBtn)
        stepsBox.addStretch(1)

        hline2 = QFrame()
        hline2.setFrameShape(QFrame.HLine)
        hline2.setFrameShadow(QFrame.Sunken)

        # Stack everything up in a vetical layout
        vbox = QVBoxLayout()
        vbox.addLayout(miniViewBox)
        vbox.addLayout(stepsBox)
        vbox.addWidget(hline2)
        vbox.addWidget(QLabel("General mission"))
        vbox.addWidget(self.missionBox)
        vbox.addLayout(buttonBox)

        return vbox

    def createButtons(self):
        """Create the row of UI buttons"""

        stepButton = QPushButton("Step")
        stepButton.clicked.connect(self.stepClicked)

        minusButton = QPushButton("- Reward")
        minusButton.clicked.connect(self.minusReward)

        plusButton = QPushButton("+ Reward")
        plusButton.clicked.connect(self.plusReward)

        slider = QSlider(Qt.Horizontal, self)
        slider.setFocusPolicy(Qt.NoFocus)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(0)
        slider.valueChanged.connect(self.setFrameRate)

        self.fpsLabel = QLabel("Manual")
        self.fpsLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.fpsLabel.setAlignment(Qt.AlignCenter)
        self.fpsLabel.setMinimumSize(80, 10)

        # Assemble the buttons into a horizontal layout
        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(stepButton)
        hbox.addWidget(slider)
        hbox.addWidget(self.fpsLabel)
        hbox.addStretch(1)
        hbox.addWidget(minusButton)
        hbox.addWidget(plusButton)
        hbox.addStretch(1)

        return hbox

    def keyPressEvent(self, e):
        actions = self.env.unwrapped.actions
        if e.key() == Qt.Key_Left:
            self.stepEnv(actions.left)
        elif e.key() == Qt.Key_Right:
            self.stepEnv(actions.right)
        elif e.key() == Qt.Key_Up:
            self.stepEnv(actions.forward)
        elif e.key() == Qt.Key_Space:
            self.stepEnv(actions.toggle)

    def mousePressEvent(self, event):
        self.clearFocus()
        QMainWindow.mousePressEvent(self, event)

    def clearFocus(self):
        """
        Clear the focus of the text boxes and buttons
        """

        # Get the object currently in focus
        focused = QApplication.focusWidget()

        if isinstance(focused, (QPushButton, QTextEdit)):
            focused.clearFocus()

    def missionEdit(self):
        # The agent will get the mission as an observation
        # before performing the next action
        text = self.missionBox.toPlainText()
        self.lastObs['mission'] = text

    def plusReward(self):
        print('+reward')
        self.env.setReward(1)

    def minusReward(self):
        print('-reward')
        self.env.setReward(-1)

    def stepClicked(self):
        self.stepEnv(action=None)

    def setFrameRate(self, value):
        """Set the frame rate limit. Zero for manual stepping."""

        #print('Set frame rate: %s' % value)

        self.fpsLimit = int(value)

        if value == 0:
            self.fpsLabel.setText("Manual")
            self.stepTimer.stop()
        else:
            value = 5
            self.fpsLabel.setText("%s FPS" % value)
            self.stepTimer.setInterval(int(1000 / value))
            self.stepTimer.start()

        self.rollout = None
        self.resetEnv()
        self.clearFocus()

    def resetBtn(self):
        self.clearFocus()
        self.resetEnv()

    def resetEnv(self):
        seed = random.randint(0, 0xFFFFFFFF)

        self.env.seed(seed)
        obs = self.env.reset()

        if not isinstance(obs, dict):
            obs = { 'image': obs, 'mission': '' }

        # If no mission is specified
        if obs['mission']:
            mission = obs['mission']
        else:
            mission = "Get to the green goal square"

        self.lastObs = obs

        self.missionBox.setPlainText(mission)

        self.showEnv(obs)

        if self.rollout and self.rollout.total_reward > 0:
            self.rollouts.append(self.rollout)
            print('num rollouts: %d' % len(self.rollouts))

        self.rollout = Rollout(seed)

    def reseedEnv(self):
        import random
        seed = random.randint(0, 0xFFFFFFFF)
        self.env.seed(seed)
        self.resetEnv()

    def train(self):
        for i in range(0, 100):
            total_loss = 0
            for r in self.rollouts:
                total_loss += train_model(self.model, r)
            print(total_loss / len(self.rollouts))

        if len(self.rollouts) < 3:
            return

        seed = random.randint(0, 0xFFFFFFFF)
        best_rollout = Rollout(0)
        num_success = 0

        for j in range(0, 100):
            rollout = run_model(self.model, self.env, seed, eps=0)

            if rollout.total_reward > best_rollout.total_reward:
                best_rollout = rollout
                num_success += 1

        print('num success: %d' % num_success)

        if best_rollout.total_reward > 0:
            self.rollouts.append(best_rollout)

        print('num rollouts: %d' % len(self.rollouts))

        self.resetEnv()

    def showEnv(self, obs):
        unwrapped = self.env.unwrapped

        # Render and display the environment
        pixmap = self.env.render(mode='pixmap')
        self.imgLabel.setPixmap(pixmap)

        # Render and display the agent's view
        image = obs['image']
        obsPixmap = unwrapped.getObsRender(image)
        self.obsImgLabel.setPixmap(obsPixmap)

        # Update the mission text
        mission = obs['mission']
        self.missionBox.setPlainText(mission)

        # Set the steps remaining
        stepsRem = unwrapped.getStepsRemaining()
        self.stepsLabel.setText(str(stepsRem))

    def stepEnv(self, action=None):
        # If the environment doesn't supply a mission, get the
        # mission from the input text box
        if not hasattr(self.lastObs, 'mission'):
            text = self.missionBox.toPlainText()
            self.lastObs['mission'] = text

        # If no manual action was specified by the user
        if action == None:
            action = self.model.select_action(self.lastObs)

        obs, reward, done, info = self.env.step(action)

        if self.rollout:
            self.rollout.append(self.lastObs, action, reward)
            print('action=%s, reward=%s' % (action, reward))

        if not isinstance(obs, dict):
            obs = { 'image': obs, 'mission': '' }

        self.showEnv(obs)
        self.lastObs = obs

        if done:
            self.resetEnv()

    def stepLoop(self):
        """Auto stepping loop, runs in its own thread"""

        print('stepLoop')

        while True:
            if self.fpsLimit == 0:
                time.sleep(0.1)
                continue

            if self.fpsLimit < 100:
                time.sleep(0.1)

            self.stepEnv()

def main(argv):
    parser = OptionParser()
    parser.add_option(
        "--env-name",
        help="gym environment to load",
        default='MiniGrid-DoorKey-8x8-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    model = Model(
        env.observation_space.spaces['image'].shape,
        env.action_space.n
    )

    # Create the application window
    app = QApplication(sys.argv)
    window = AIGameWindow(env, model)

    # Run the application
    sys.exit(app.exec_())

if __name__ == '__main__':
    main(sys.argv)
