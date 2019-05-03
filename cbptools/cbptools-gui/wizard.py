# !/usr/bin/env python
import tkinter as tk
import nibabel as nib
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os

font = 'fangsong ti'


class TitleFrame(tk.Frame):
    def __init__(self, master, title, width):
        super().__init__(master)
        self.config(width=width, height=52)

        title_frame = tk.Frame(self, width=width, height=50, bg='white')
        title_frame.place(x=0, y=0)
        title_frame.grid_propagate(False)
        tk.Frame(self, width=width, height=1, bg='#9fa19e').place(x=0, y=50)
        tk.Frame(self, width=width, height=1, bg='#ffffff').place(x=0, y=51)

        tk.Message(
            title_frame,
            text=title,
            width=width,
            font=(font, 12, 'bold'),
            bg='white'
        ).place(x=5, y=15)


class Pages:
    @staticmethod
    def display() -> bool:
        """Display condition for the step. Returns False if not met"""
        return True

    @staticmethod
    def validate() -> bool:
        """Validation of all input parameters for the current step"""
        return True

    @staticmethod
    def get_values() -> dict:
        """Get all input parameters in a dictionary"""
        return {}


class Page1(tk.Frame, Pages):
    def __init__(self, master, width, height):
        super().__init__(master)
        self.config(width=width, height=height)
        self.master = master

        banner_frame = tk.Frame(self, width=150, height=height, bg='blue')
        banner_frame.place(x=0, y=0)
        content_frame = tk.Frame(self, width=width-150, height=150, pady=5, padx=5)
        content_frame.place(x=150, y=0)

        tk.Message(
            content_frame,
            text="Welcome to the CBPtools Project Setup Wizard",
            width=width-160,
            font=(font, 12, 'bold')
        ).grid(column=0, row=0, sticky='nw', pady=5)

        tk.Message(
            content_frame,
            text='This will create a project directory for connectivity-based parcellation '
                 'and install the workflow into that directory with the given parameters.',
            width=width-160,
            font=(font, 10)
        ).grid(column=0, row=1, sticky='nw', pady=5)

        tk.Message(
            content_frame,
            text='Click Next to continue, or Cancel to exit Setup.',
            width=width-160,
            font=(font, 10)
        ).grid(column=0, row=2, sticky='nw')


class Page2(tk.Frame, Pages):
    def __init__(self, master, width, height):
        super().__init__(master)
        self.config(width=width, height=height)
        self.master = master

        # Step Title
        TitleFrame(self, title='Select Project Folder', width=width).place(x=0, y=0)

        # Description Frame
        text_frame = tk.Frame(self, width=width, height=100, pady=5, padx=5)
        text_frame.place(x=0, y=52)
        tk.Message(
            text_frame,
            text='This wizard will set up a CBPtools project on your computer.',
            width=width,
            font=(font, 10)
        ).grid(column=0, row=0, sticky='nw')
        tk.Message(
            text_frame,
            text='Select a folder on your system by clicking "Browse" to create the project folder at the specified '
                 'location.',
            width=width,
            font=(font, 10)
        ).grid(column=0, row=1, sticky='nw', pady=10)

        # Content Frame
        content_frame = tk.Frame(self, width=width, height=height - 202, pady=15, padx=15)
        content_frame.place(x=0, y=152)

        self.work_dir = tk.StringVar(value=os.path.join(str(Path.home()), 'cbptools_project'))
        tk.Label(content_frame, text='Folder:').grid(column=0, row=0, sticky='nw', padx=15)
        frame_workdir = tk.Frame(content_frame, borderwidth=1, relief='sunken')
        frame_workdir.grid(column=0, row=1, padx=15)
        self.entry_workdir = tk.Entry(
            frame_workdir,
            width=62,
            borderwidth=3,
            relief='flat',
            textvariable=self.work_dir
        )
        self.entry_workdir.grid(column=0, row=1)
        tk.Button(
            content_frame,
            text='Browse',
            width=15,
            command=lambda: self.askdirectory(self.entry_workdir)
        ).grid(column=1, row=1)

        self.force_overwrite = tk.BooleanVar(value=False)
        tk.Checkbutton(
            content_frame,
            text=' Force Overwrite',
            variable=self.force_overwrite
        ).grid(column=0, row=2, padx=15, pady=10, sticky='w')

    @staticmethod
    def askdirectory(e):
        home = str(Path.home())
        text = 'Please select a directory to store the project in'
        folder = filedialog.askdirectory(initialdir=home, title=text)
        if folder is not None:
            e.delete(0, tk.END)
            e.insert(0, folder)

    def validate(self) -> bool:
        """Validate input of the current step"""
        valid = {'work_dir': True}
        work_dir = self.work_dir.get()

        # Validate work_dir
        if work_dir == '':
            valid['work_dir'] = False

        elif os.path.exists(work_dir) and len(os.listdir(work_dir)) > 0 and not self.force_overwrite.get():
            valid['work_dir'] = False

        # Set view
        self.entry_workdir.config({'background': '#ffffff'})
        if valid['work_dir'] is False:
            self.entry_workdir.config({'background': '#eaaeae'})

        return not any(value is False for value in valid.values())

    def get_values(self):
        parameters = {
            'work_dir': self.work_dir.get(),
            'force_overwrite': self.force_overwrite.get()
        }
        return parameters


class Page3(tk.Frame, Pages):
    """Select DataSet"""

    def __init__(self, master, width, height):
        super().__init__(master)
        self.config(width=width, height=height)

        TitleFrame(self, title='Select DataSet', width=width).grid(column=0, row=0)

        # Text
        frame_description = tk.Frame(self, width=width)
        frame_description.place(x=0, y=52)
        tk.Message(
            frame_description,
            text='Define the path to the dataset folder and list all participants to be included in this project.',
            width=width,
            font=(font, 10)
        ).grid(column=0, row=0, sticky='nw', padx=5, pady=5)

        # Modality
        frame_modality = tk.Frame(self)
        frame_modality.grid(column=0, row=1, pady=(55, 5), padx=20, sticky='nw')
        self.modality = tk.StringVar(value='fmri')
        tk.Label(frame_modality, text='Modality:').grid(column=0, row=0, padx=5)
        tk.Radiobutton(
            frame_modality,
            text=' Resting-State fMRI',
            value='fmri',
            variable=self.modality
        ).grid(column=1, row=0, sticky='nw')
        tk.Radiobutton(
            frame_modality,
            text=' Diffusion MRI',
            value='dmri',
            variable=self.modality
        ).grid(column=2, row=0, sticky='nw')

        # Dataset Path
        frame_dataset = tk.Frame(self)
        frame_dataset.grid(column=0, row=2, pady=(5, 5), padx=20, sticky='nw')

        self.dataset = tk.StringVar(value=os.path.join(str(Path.home())))
        tk.Label(frame_dataset, text='Path to DataSet:').grid(column=0, row=1, sticky='nw', padx=5)

        frame_entry_dataset = tk.Frame(frame_dataset, borderwidth=1, relief='sunken')
        frame_entry_dataset.grid(column=0, row=2, padx=5)
        entry_dataset = tk.Entry(
            frame_entry_dataset,
            width=60,
            borderwidth=3,
            relief='flat',
            textvariable=self.dataset
        )
        entry_dataset.grid(column=0, row=2, sticky='nw')
        tk.Button(
            frame_dataset,
            text='Browse',
            width=15,
            command=lambda: self.askdirectory(entry_dataset)
        ).grid(column=1, row=2, sticky='nw', padx=(0, 10))

        # Participants File
        tk.Label(self, text='Participants:').grid(column=0, row=3, sticky='nw', padx=25, pady=(5, 0))
        frame_participants = tk.Frame(self, borderwidth=1, relief='sunken')
        frame_participants.grid(column=0, row=4, padx=(25, 10), pady=5, sticky='nw')

        self.auto_detect = tk.IntVar()
        tk.Checkbutton(
            frame_participants,
            text=' Auto-Detect Participants',
            variable=self.auto_detect,
            command=self.autodetect
        ).grid(column=0, row=0, sticky='nw', padx=5, pady=(5, 10))

        self.participants = tk.StringVar(value=os.path.join(str(Path.home())))
        tk.Label(frame_participants, text='Path to Participants File:').grid(column=0, row=1, sticky='nw', padx=10)
        frame_entry_participants = tk.Frame(frame_participants, borderwidth=1, relief='sunken')
        frame_entry_participants.grid(column=0, row=2, padx=10)
        self.entry_participants = tk.Entry(
            frame_entry_participants,
            width=58,
            borderwidth=3,
            relief='flat',
            textvariable=self.participants
        )
        self.entry_participants.grid(column=0, row=0, sticky='nw')
        self.participants_button = tk.Button(
            frame_participants,
            text='Browse',
            width=15,
            command=lambda: self.askfile(self.entry_participants)
        )
        self.participants_button.grid(column=1, row=2, sticky='nw', padx=(0, 10))

        # Frame for Participant Options
        frame_options_participants = tk.Frame(frame_participants)
        frame_options_participants.grid(column=0, row=3, padx=10, sticky='nw')

        tk.Label(
            frame_options_participants,
            text='File Separator:'
        ).grid(column=0, row=0, sticky='nw', pady=(10, 0), padx=(0, 20))
        self.confounds_sep = ttk.Combobox(
            frame_options_participants,
            values=['\\t (.tsv)', ', (.csv)', '; (.xls)'],
            state='readonly',
            width=15
        )
        self.confounds_sep.grid(column=0, row=1, sticky='nw', pady=10)
        ttk.Style().configure('TCombobox', padding=5, arrowsize=15, background='#ffffff', relief='flat')
        ttk.Style().map('TCombobox', fieldbackground=[('readonly', 'white')])
        ttk.Style().map('TCombobox', selectbackground=[('readonly', 'white')])
        ttk.Style().map('TCombobox', selectforeground=[('readonly', 'black')])

        tk.Label(
            frame_options_participants,
            text='Subject ID Column:'
        ).grid(column=1, row=0, sticky='nw', pady=(10, 0), padx=(20, 10))

        self.confounds_col = tk.StringVar(value='participant_id')
        frame_entry_confounds = tk.Frame(frame_options_participants, borderwidth=1, relief='sunken')
        frame_entry_confounds.grid(column=1, row=1, padx=(20, 10))
        self.confounds_col_entry = tk.Entry(
            frame_entry_confounds,
            width=25,
            borderwidth=3,
            relief='flat',
            textvariable=self.confounds_col
        )
        self.confounds_col_entry.grid(column=1, row=1, sticky='nw')

    @staticmethod
    def askdirectory(e):
        home = str(Path.home())
        text = 'Please select the full path to the dataset'
        folder = filedialog.askdirectory(initialdir=home, title=text)
        if folder is not None and folder:
            e.delete(0, tk.END)
            e.insert(0, folder)

    @staticmethod
    def askfile(e):
        home = str(Path.home())
        text = 'Please select the full path to the participants file'
        file = filedialog.askopenfilename(initialdir=home, title=text)
        if file is not None and file:
            e.delete(0, tk.END)
            e.insert(0, file)

    def autodetect(self):
        if self.auto_detect.get() == 1:
            self.entry_participants.config(state='disabled')
            self.participants_button.config(state='disabled')
            self.confounds_sep.config(state='disabled')
            self.confounds_col_entry.config(state='disabled')
        else:
            self.entry_participants.config(state='normal')
            self.participants_button.config(state='normal')
            self.confounds_sep.config(state='readonly')
            self.confounds_col_entry.config(state='normal')

    def validate(self) -> bool:
        """Validate input of the current step"""
        valid = {'modality': True}
        modality = self.modality.get()

        # Validate modality
        if modality != 'fmri' and modality != 'dmri':
            valid['modality'] = False

        return not any(value is False for value in valid.values())

    def get_values(self):
        parameters = {
            'modality': self.modality.get()
        }
        return parameters


class Page4RSFMRI(tk.Frame, Pages):
    """Select rs-fMRI DataSet"""

    def __init__(self, master, width, height):
        super().__init__(master)
        self.config(width=width, height=height)

        TitleFrame(self, title='Select rs-fMRI Data', width=width).place(x=0, y=0)

        frame_description = tk.Frame(self, width=width, height=150, pady=5, padx=5)
        frame_description.grid(column=0, row=0, pady=(52, 0))
        tk.Message(
            frame_description,
            text='When defining data file paths, replace the subject identifier (defined in the previous step) with '
                 'the wildcard {participant_id}. The wildcard will be replaced by the identifier.',
            width=width-10,
            font=(font, 10)
        ).grid(column=0, row=0, sticky='nw', padx=(0, 5))
        tk.Message(
            frame_description,
            text='Example: "/path/to/{participant_id}/file.nii" will become "/path/to/sub-001/file.nii" for the '
                 'participant with ID "sub-001".',
            width=width,
            font=(font, 10)
        ).grid(column=0, row=1, sticky='nw', padx=(0, 5), pady=10)
        tk.Message(
            frame_description,
            text='All entered paths are relative to the DataSet path defined in the previous step.',
            width=width,
            font=(font, 10)
        ).grid(column=0, row=2, sticky='nw', padx=(0, 5), pady=10)

        frame_files = tk.Frame(self)
        frame_files.grid(column=0, row=1, pady=(10, 0), padx=10, sticky='nw')
        self.time_series = tk.StringVar()
        tk.Label(
            frame_files,
            text='Relative Path to Time-Series (.nii):'
        ).grid(column=0, row=0, sticky='nw', padx=15, pady=(10, 0))
        frame_timeseries = tk.Frame(frame_files, borderwidth=1, relief='sunken')
        frame_timeseries.grid(column=0, row=1, padx=15)
        tk.Entry(
            frame_timeseries,
            width=62,
            borderwidth=3,
            relief='flat'
        ).grid(column=0, row=1)

        self.confounds = tk.StringVar()
        tk.Label(
            frame_files,
            text='Relative Path to Confounds (optional):'
        ).grid(column=0, row=2, sticky='nw', padx=15, pady=(10, 0))
        frame_confounds = tk.Frame(frame_files, borderwidth=1, relief='sunken')
        frame_confounds.grid(column=0, row=3, padx=15)
        tk.Entry(
            frame_confounds,
            width=62,
            borderwidth=3,
            relief='flat'
        ).grid(column=0, row=1)
        tk.Label(
            frame_files,
            text='File Separator:'
        ).grid(column=1, row=2, sticky='nw', pady=(10, 0))
        self.confounds_sep = ttk.Combobox(
            frame_files,
            values=['\\t (.tsv)', ', (.csv)', '; (.xls)'],
            state='readonly',
            width=10
        )
        self.confounds_sep.grid(column=1, row=3, sticky='nw')
        ttk.Style().configure('TCombobox', padding=5, arrowsize=15, background='#ffffff', relief='flat')
        ttk.Style().map('TCombobox', fieldbackground=[('readonly', 'white')])
        ttk.Style().map('TCombobox', selectbackground=[('readonly', 'white')])
        ttk.Style().map('TCombobox', selectforeground=[('readonly', 'black')])

    def display(self):
        return self.master.parameters['modality'] == 'fmri'


class Page4DMRI(tk.Frame, Pages):
    """Select dMRI DataSet"""

    def __init__(self, master, width, height):
        super().__init__(master)
        self.config(width=width, height=height)

        TitleFrame(self, title='Select dMRI Data', width=width).place(x=0, y=0)
        text_frame = tk.Frame(self, width=width, height=100, pady=5, padx=5)
        text_frame.place(x=0, y=52)
        content_frame = tk.Frame(self, width=width, height=height - 202, pady=15, padx=15)
        content_frame.place(x=0, y=152)

        tk.Message(
            text_frame,
            text='Define the path to the dataset folder.',
            width=width,
            font=(font, 10)
        ).grid(column=0, row=0, sticky='nw')
        tk.Message(
            text_frame,
            text='For the time-series and confounds paths, use {participant_id} to replace the subject identifier. '
                 'This wildcard will be replaced by the actual participant ID (e.g. "sub-001") during processing.',
            width=width,
            font=(font, 10)
        ).grid(column=0, row=1, sticky='nw', pady=10)

        self.bet_binary_mask = tk.StringVar(value=str(Path.home()))
        tk.Label(content_frame, text='BET Binary Mask:').grid(column=0, row=0, sticky='nw', padx=15)
        frame_bet_binary_mask = tk.Frame(content_frame, borderwidth=1, relief='sunken')
        frame_bet_binary_mask.grid(column=0, row=1, padx=15, sticky='nw')
        tk.Entry(
            frame_bet_binary_mask,
            width=75,
            borderwidth=3,
            relief='flat',
            textvariable=self.bet_binary_mask
        ).grid(column=0, row=1, sticky='nw')

        self.xfm = tk.StringVar(value=str(Path.home()))
        tk.Label(
            content_frame,
            text='Transform taking seed space to DTI space (FLIRT matrix or FNIR warpfield):'
        ).grid(column=0, row=2, sticky='nw', padx=15, pady=(10, 0))
        frame_xfm = tk.Frame(content_frame, borderwidth=1, relief='sunken')
        frame_xfm.grid(column=0, row=3, padx=15, sticky='nw')
        tk.Entry(
            frame_xfm,
            width=75,
            borderwidth=3,
            relief='flat',
            textvariable=self.xfm
        ).grid(column=0, row=1, sticky='nw')

        self.inv_xfm = tk.StringVar(value=str(Path.home()))
        tk.Label(
            content_frame,
            text='Transform taking DTI space to seed space:'
        ).grid(column=0, row=4, sticky='nw', padx=15, pady=(10, 0))
        frame_inv_xfm = tk.Frame(content_frame, borderwidth=1, relief='sunken')
        frame_inv_xfm.grid(column=0, row=5, padx=15, sticky='nw')
        tk.Entry(
            frame_inv_xfm,
            width=75,
            borderwidth=3,
            relief='flat',
            textvariable=self.inv_xfm
        ).grid(column=0, row=1, sticky='nw')

        self.samples = tk.StringVar(value=str(Path.home()))
        tk.Label(
            content_frame,
            text='Path and basename for samples files:'
        ).grid(column=0, row=6, sticky='nw', padx=15, pady=(10, 0))
        frame_samples = tk.Frame(content_frame, borderwidth=1, relief='sunken')
        frame_samples.grid(column=0, row=7, padx=15, sticky='nw')
        tk.Entry(
            frame_samples,
            width=75,
            borderwidth=3,
            relief='flat',
            textvariable=self.samples
        ).grid(column=0, row=1, sticky='nw')

    def display(self):
        return self.master.parameters['modality'] == 'dmri'

    def validate(self):
        valid = {
            'bet_binary_mask': True,
            'xfm': True,
            'inv_xfm': True,
            'samples': True
        }

        bet_binary_mask = self.bet_binary_mask.get()
        xfm = self.xfm.get()
        inv_xfm = self.inv_xfm.get()
        samples = self.samples.get()

        # TODO: Validate input (check for {participant_id})

        return not any(value is False for value in valid.values())


class Page5(tk.Frame, Pages):
    def __init__(self, master, width, height):
        super().__init__(master)
        self.config(width=width, height=height)

        TitleFrame(self, title='Select Confounds to include', width=width).place(x=0, y=0)

    def display(self):
        # TODO: Display only if a confounds file was added
        return self.master.parameters['modality'] == 'fmri'


class Page6RSFMRI(tk.Frame, Pages):
    def __init__(self, master, width, height):
        super().__init__(master)
        self.config(width=width, height=height)

        TitleFrame(self, title='rs-fMRI Seed Mask', width=width).grid(column=0, row=0)

        # Text
        frame_description = tk.Frame(self, width=width)
        frame_description.place(x=0, y=52)
        tk.Message(
            frame_description,
            text='Select the seed mask file and its preprocessing options. Note that the seed image must be in the '
                 'same space as the target mask and time-series images.',
            width=width,
            font=(font, 10)
        ).grid(column=0, row=0, sticky='nw', padx=5, pady=5)

        # Content
        frame_content = tk.Frame(self, width=width)
        frame_content.grid(column=0, row=1, sticky='nw', padx=10, pady=(50, 0))

        # Image Canvas
        frame_img = tk.Frame(frame_content, borderwidth=1, relief='sunken')
        frame_img.grid(column=0, row=0, padx=(10, 10), pady=5, sticky='nw')
        fig, ax = plt.subplots(figsize=(2.9, 1.5))
        fig.patch.set_facecolor('black')
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        canvas = FigureCanvasTkAgg(fig, master=frame_img)
        canvas.get_tk_widget().grid(column=0, row=0)
        canvas.draw()

        # Mask
        frame_mask = tk.Frame(frame_content, borderwidth=1, relief='sunken')
        frame_mask.grid(column=0, row=1, padx=(10, 10), pady=5, sticky='nw')
        frame_mask.columnconfigure(0, minsize=(width / 2) - 35)
        self.label_filename = tk.Message(
            frame_mask,
            text='Filename: None',
            width=275,
            font=(font, 10),
        )
        self.label_filename.grid(column=0, row=1, sticky='nw', padx=2, pady=2)
        self.label_shape = tk.Message(
            frame_mask,
            text='Shape: None',
            width=275,
            font=(font, 10)
        )
        self.label_shape.grid(column=0, row=3, sticky='nw', padx=2, pady=2)
        self.label_voxel_size = tk.Message(
            frame_mask,
            text='Voxel Size: None',
            width=275,
            font=(font, 10)
        )
        self.label_voxel_size.grid(column=0, row=4, sticky='nw', padx=2, pady=2)
        self.label_origin = tk.Message(
            frame_mask,
            text='Origin: None',
            width=275,
            font=(font, 10)
        )
        self.label_origin.grid(column=0, row=5, sticky='nw', padx=2, pady=2)

        self.target_button = tk.Button(
            frame_mask,
            text='Seed Mask File (.nii/.nii.gz)',
            width=35,
            command=lambda: self.askfile(
                filename=self.label_filename,
                shape=self.label_shape,
                voxel_size=self.label_voxel_size,
                origin=self.label_origin,
                ax=ax,
                canvas=canvas
            )
        )
        self.target_button.grid(column=0, row=0, padx=(5, 5), pady=5)

        # Options
        frame_options = tk.Frame(frame_content, width=width)
        frame_options.grid(column=1, row=0, sticky='nw', padx=10, pady=0)

        self.median_filter = tk.BooleanVar(value=False)
        tk.Checkbutton(
            frame_options,
            text=' Apply Median Filter',
            variable=self.median_filter
        ).grid(column=0, row=2, padx=15, pady=10, sticky='w')

    @staticmethod
    def askfile(filename, shape, voxel_size, origin, ax, canvas):
        home = str(Path.home())
        text = 'Please select the NIfTI (.nii or .nii.gz) image of the seed mask'
        file = filedialog.askopenfilename(
            initialdir=home,
            title=text,
            filetypes=(("NIfTI files", "*.nii"), ("GZipped NIfTI files", "*.nii.gz"))
        )

        if file is not None and file:
            try:
                img = nib.load(file)
                v = nib.affines.voxel_sizes(img.affine)
                o = img.affine[0:3, 3].astype(int)
                filename.configure(text=f'Filename: {os.path.basename(file)}')
                shape.configure(text=f'Shape: {img.shape}')
                voxel_size.configure(text=f'Voxel Size: {v[0]} x {v[1]} x {v[2]}')
                origin.configure(text=f'Origin: x={o[0]}, y={o[1]}, z={o[2]}')
                data = img.get_data()
                ax.imshow(data[:, :, o[2].astype(int)//2], cmap="gray", origin="lower")
                canvas.draw()
            except:
                filename.configure(text='Filename: None')
                shape.configure(text='Shape: None')
                voxel_size.configure(text='Voxel Size: None')
                origin.configure(text='Origin: None')

    def display(self):
        return self.master.parameters['modality'] == 'fmri'


class Page6DMRI(tk.Frame, Pages):
    def __init__(self, master, width, height):
        super().__init__(master)
        self.config(width=width, height=height)

        TitleFrame(self, title='dMRI Seed Mask', width=width).grid(column=0, row=0)

        # Text
        frame_description = tk.Frame(self, width=width)
        frame_description.place(x=0, y=52)
        tk.Message(
            frame_description,
            text='Select the seed mask file and its preprocessing options. Note that the seed image must be in the '
                 'same space as the target mask and the bedpostX processed input data.',
            width=width,
            font=(font, 10)
        ).grid(column=0, row=0, sticky='nw', padx=5, pady=5)

        # Content
        frame_content = tk.Frame(self, width=width)
        frame_content.grid(column=0, row=1, sticky='nw', padx=10, pady=(50, 0))

        # Image Canvas
        frame_img = tk.Frame(frame_content, borderwidth=1, relief='sunken')
        frame_img.grid(column=0, row=0, padx=(10, 10), pady=5, sticky='nw')
        fig, ax = plt.subplots(figsize=(2.9, 1.5))
        fig.patch.set_facecolor('black')
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        canvas = FigureCanvasTkAgg(fig, master=frame_img)
        canvas.get_tk_widget().grid(column=0, row=0)
        canvas.draw()

        # Mask
        frame_mask = tk.Frame(frame_content, borderwidth=1, relief='sunken')
        frame_mask.grid(column=0, row=1, padx=(10, 10), pady=5, sticky='nw')
        frame_mask.columnconfigure(0, minsize=(width / 2) - 35)
        self.label_filename = tk.Message(
            frame_mask,
            text='Filename: None',
            width=275,
            font=(font, 10),
        )
        self.label_filename.grid(column=0, row=1, sticky='nw', padx=2, pady=2)
        self.label_shape = tk.Message(
            frame_mask,
            text='Shape: None',
            width=275,
            font=(font, 10)
        )
        self.label_shape.grid(column=0, row=3, sticky='nw', padx=2, pady=2)
        self.label_voxel_size = tk.Message(
            frame_mask,
            text='Voxel Size: None',
            width=275,
            font=(font, 10)
        )
        self.label_voxel_size.grid(column=0, row=4, sticky='nw', padx=2, pady=2)
        self.label_origin = tk.Message(
            frame_mask,
            text='Origin: None',
            width=275,
            font=(font, 10)
        )
        self.label_origin.grid(column=0, row=5, sticky='nw', padx=2, pady=2)

        self.mask_button = tk.Button(
            frame_mask,
            text='Seed Mask File (.nii/.nii.gz)',
            width=35,
            command=lambda: self.askfile(
                filename=self.label_filename,
                shape=self.label_shape,
                voxel_size=self.label_voxel_size,
                origin=self.label_origin,
                ax=ax,
                canvas=canvas
            )
        )
        self.mask_button.grid(column=0, row=0, padx=(5, 5), pady=5)

        # Options
        frame_options = tk.Frame(frame_content, width=width)
        frame_options.grid(column=1, row=0, sticky='nw', padx=10, pady=0)

        self.median_filter = tk.BooleanVar(value=False)
        tk.Checkbutton(
            frame_options,
            text=' Apply Median Filter',
            variable=self.median_filter
        ).grid(column=0, row=0, padx=15, pady=10, sticky='w')

        self.upsample = tk.BooleanVar(value=False)
        tk.Checkbutton(
            frame_options,
            text=' Upsample Seed Mask',
            variable=self.upsample,
            command=self.change_state,
        ).grid(column=0, row=1, padx=15, pady=10, sticky='w')

        frame_upsample = tk.Frame(frame_options)
        frame_upsample.grid(column=0, row=2, padx=40, pady=0, sticky='nw')

        tk.Label(frame_upsample, text='X:').grid(column=0, row=0, sticky='nw', padx=2, pady=2)
        self.upsample_x = tk.DoubleVar()
        self.entry_upsample_x = tk.Entry(
            frame_upsample,
            width=3,
            borderwidth=3,
            relief='flat',
            state='disabled',
            textvariable=self.upsample_x
        )
        self.entry_upsample_x.grid(column=1, row=0, sticky='nw')

        tk.Label(frame_upsample, text='Y:').grid(column=2, row=0, sticky='nw', padx=2, pady=2)
        self.upsample_y = tk.DoubleVar()
        self.entry_upsample_y = tk.Entry(
            frame_upsample,
            width=3,
            borderwidth=3,
            relief='flat',
            state='disabled',
            textvariable=self.upsample_y
        )
        self.entry_upsample_y.grid(column=3, row=0, sticky='nw')

        tk.Label(frame_upsample, text='Z:').grid(column=4, row=0, sticky='nw', padx=2, pady=2)
        self.upsample_z = tk.DoubleVar()
        self.entry_upsample_z = tk.Entry(
            frame_upsample,
            width=3,
            borderwidth=3,
            relief='flat',
            state='disabled',
            textvariable=self.upsample_z
        )
        self.entry_upsample_z.grid(column=5, row=0, sticky='nw')

    @staticmethod
    def askfile(filename, shape, voxel_size, origin, ax, canvas):
        home = str(Path.home())
        text = 'Please select the NIfTI (.nii or .nii.gz) image of the seed mask'
        file = filedialog.askopenfilename(
            initialdir=home,
            title=text,
            filetypes=(("NIfTI files", "*.nii"), ("GZipped NIfTI files", "*.nii.gz"))
        )

        if file is not None and file:
            try:
                img = nib.load(file)
                v = nib.affines.voxel_sizes(img.affine)
                o = img.affine[0:3, 3].astype(int)
                filename.configure(text=f'Filename: {os.path.basename(file)}')
                shape.configure(text=f'Shape: {img.shape}')
                voxel_size.configure(text=f'Voxel Size: {v[0]} x {v[1]} x {v[2]}')
                origin.configure(text=f'Origin: x={o[0]}, y={o[1]}, z={o[2]}')
                data = img.get_data()
                ax.imshow(data[:, :, o[2].astype(int)//2], cmap="gray", origin="lower")
                canvas.draw()
            except:
                filename.configure(text='Filename: None')
                shape.configure(text='Shape: None')
                voxel_size.configure(text='Voxel Size: None')
                origin.configure(text='Origin: None')

    def change_state(self):
        if self.upsample.get() == 1:
            self.entry_upsample_x.config(state='normal')
            self.entry_upsample_y.config(state='normal')
            self.entry_upsample_z.config(state='normal')
        else:
            self.entry_upsample_x.config(state='disabled')
            self.entry_upsample_y.config(state='disabled')
            self.entry_upsample_z.config(state='disabled')

    def display(self):
        return self.master.parameters['modality'] == 'dmri'


class Page7RSFMRI(tk.Frame, Pages):
    def __init__(self, master, width, height):
        super().__init__(master)
        self.config(width=width, height=height)

        TitleFrame(self, title='rs-fMRI Target Mask', width=width).grid(column=0, row=0)

        # Text
        frame_description = tk.Frame(self, width=width)
        frame_description.place(x=0, y=52)
        tk.Message(
            frame_description,
            text='Select the target mask file and its preprocessing options. Note that the target image must be in the '
                 'same space as the seed mask and time-series images.',
            width=width,
            font=(font, 10)
        ).grid(column=0, row=0, sticky='nw', padx=5, pady=5)

        # Content
        frame_content = tk.Frame(self, width=width)
        frame_content.grid(column=0, row=1, sticky='nw', padx=10, pady=(50, 0))

        # Image Canvas
        frame_img = tk.Frame(frame_content, borderwidth=1, relief='sunken')
        frame_img.grid(column=0, row=0, padx=(10, 10), pady=5, sticky='nw')
        fig, ax = plt.subplots(figsize=(2.9, 1.5))
        fig.patch.set_facecolor('black')
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        canvas = FigureCanvasTkAgg(fig, master=frame_img)
        canvas.get_tk_widget().grid(column=0, row=0)
        canvas.draw()

        # Mask
        frame_mask = tk.Frame(frame_content, borderwidth=1, relief='sunken')
        frame_mask.grid(column=0, row=1, padx=(10, 10), pady=5, sticky='nw')
        frame_mask.columnconfigure(0, minsize=(width / 2) - 35)
        self.label_filename = tk.Message(
            frame_mask,
            text='Filename: None',
            width=275,
            font=(font, 10),
        )
        self.label_filename.grid(column=0, row=1, sticky='nw', padx=2, pady=2)
        self.label_shape = tk.Message(
            frame_mask,
            text='Shape: None',
            width=275,
            font=(font, 10)
        )
        self.label_shape.grid(column=0, row=3, sticky='nw', padx=2, pady=2)
        self.label_voxel_size = tk.Message(
            frame_mask,
            text='Voxel Size: None',
            width=275,
            font=(font, 10)
        )
        self.label_voxel_size.grid(column=0, row=4, sticky='nw', padx=2, pady=2)
        self.label_origin = tk.Message(
            frame_mask,
            text='Origin: None',
            width=275,
            font=(font, 10)
        )
        self.label_origin.grid(column=0, row=5, sticky='nw', padx=2, pady=2)

        self.target_button = tk.Button(
            frame_mask,
            text='Target Mask File (.nii/.nii.gz)',
            width=35,
            command=lambda: self.askfile(
                filename=self.label_filename,
                shape=self.label_shape,
                voxel_size=self.label_voxel_size,
                origin=self.label_origin,
                ax=ax,
                canvas=canvas
            )
        )
        self.target_button.grid(column=0, row=0, padx=(5, 5), pady=5)

        # Options
        frame_options = tk.Frame(frame_content, width=width)
        frame_options.grid(column=1, row=0, sticky='nw', padx=10, pady=0)

        self.resample_to_mni = tk.BooleanVar(value=False)
        tk.Checkbutton(
            frame_options,
            text=' Resample to MNI152 Space',
            variable=self.resample_to_mni
        ).grid(column=0, row=0, padx=15, pady=10, sticky='w')

        self.subsample = tk.BooleanVar(value=False)
        tk.Checkbutton(
            frame_options,
            text=' Subsample',
            variable=self.subsample
        ).grid(column=0, row=1, padx=15, pady=10, sticky='w')

        self.del_seed_from_target = tk.BooleanVar(value=False)
        tk.Checkbutton(
            frame_options,
            text=' Remove seed voxels from target image',
            variable=self.del_seed_from_target,
            command=self.change_state
        ).grid(column=0, row=2, padx=15, pady=10, sticky='w')

        frame_del_seed_expand = tk.Frame(frame_options)
        frame_del_seed_expand.grid(column=0, row=3, padx=40, pady=0, sticky='nw')

        self.label_del_seed_expand1 = tk.Label(frame_del_seed_expand, text='Expand border by ', fg="gray")
        self.label_del_seed_expand1.grid(column=0, row=0, sticky='nw', padx=0, pady=0)
        self.del_seed_expand = tk.IntVar()
        self.entry_del_seed_expand = tk.Entry(
            frame_del_seed_expand,
            width=2,
            borderwidth=3,
            relief='flat',
            state='disabled',
            textvariable=self.del_seed_expand
        )
        self.entry_del_seed_expand.grid(column=1, row=0, sticky='nw')
        self.label_del_seed_expand2 = tk.Label(frame_del_seed_expand, text=' milimeter', fg='gray')
        self.label_del_seed_expand2.grid(column=2, row=0, sticky='nw', padx=0, pady=0)

    @staticmethod
    def askfile(filename, shape, voxel_size, origin, ax, canvas):
        home = str(Path.home())
        text = 'Please select the NIfTI (.nii or .nii.gz) image of the seed mask'
        file = filedialog.askopenfilename(
            initialdir=home,
            title=text,
            filetypes=(("NIfTI files", "*.nii"), ("GZipped NIfTI files", "*.nii.gz"))
        )

        if file is not None and file:
            try:
                img = nib.load(file)
                v = nib.affines.voxel_sizes(img.affine)
                o = img.affine[0:3, 3].astype(int)
                filename.configure(text=f'Filename: {os.path.basename(file)}')
                shape.configure(text=f'Shape: {img.shape}')
                voxel_size.configure(text=f'Voxel Size: {v[0]} x {v[1]} x {v[2]}')
                origin.configure(text=f'Origin: x={o[0]}, y={o[1]}, z={o[2]}')
                data = img.get_data()
                ax.imshow(data[:, :, o[2].astype(int)//2], cmap="gray", origin="lower")
                canvas.draw()
            except:
                filename.configure(text='Filename: None')
                shape.configure(text='Shape: None')
                voxel_size.configure(text='Voxel Size: None')
                origin.configure(text='Origin: None')

    def change_state(self):
        if self.del_seed_from_target.get() == 1:
            self.entry_del_seed_expand.config(state='normal')
            self.label_del_seed_expand1.config(fg='black')
            self.label_del_seed_expand2.config(fg='black')
        else:
            self.entry_del_seed_expand.config(state='disabled')
            self.label_del_seed_expand1.config(fg='gray')
            self.label_del_seed_expand2.config(fg='gray')

    def display(self):
        return self.master.parameters['modality'] == 'fmri'




class Page8RSFMRI(tk.Frame, Pages):
    def __init__(self, master, width, height):
        super().__init__(master)
        self.config(width=width, height=height)

        TitleFrame(self, title='Connectivity Options', width=width).place(x=0, y=0)

    def display(self):
        return self.master.parameters['modality'] == 'dmri'


class Page10(tk.Frame, Pages):
    def __init__(self, master, width, height):
        super().__init__(master)
        self.config(width=width, height=height)

        TitleFrame(self, title='Setup Completed', width=width).place(x=0, y=0)


class Wizard(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        tk.Frame.__init__(self, master, *args, **kwargs)

        # Parameter Storage
        self.parameters = dict()

        # Define master window properties
        self.master = master
        self.master.title('cbptools project setup')
        self.master.protocol("WM_DELETE_WINDOW", self.cancel)

        # Set up the window size and position
        width, height = (650, 450)
        sw = self.master.winfo_screenwidth()
        sh = self.master.winfo_screenheight()
        x = int((sw-width)/2)
        y = int((sh-height)/2)
        self.master.geometry(f'{width}x{height}+{x}+{y}')

        # Step tracking
        self.current_step = None
        self.steps = [
            Page1(self, width=width, height=height),
            Page2(self, width=width, height=height),
            Page3(self, width=width, height=height),
            Page4RSFMRI(self, width=width, height=height),
            Page4DMRI(self, width=width, height=height),
            Page5(self, width=width, height=height),
            Page6RSFMRI(self, width=width, height=height),
            Page6DMRI(self, width=width, height=height),
            Page7RSFMRI(self, width=width, height=height),
            Page10(self, width=width, height=height)
        ]

        # Frames
        tk.Frame(self, width=width, height=1, bg='#9fa19e').place(x=0, y=height-37)
        tk.Frame(self, width=width, height=1, bg='#ffffff').place(x=0, y=height-36)
        self.button_frame = tk.Frame(self, width=width, height=35, pady=5, padx=5)
        self.button_frame.place(x=0, y=height-35)

        # Buttons for button_frame
        self.cancel_button = tk.Button(self.button_frame, text='Cancel', command=self.cancel)
        self.cancel_button.place(x=width - 240, y=0)
        self.back_button = tk.Button(self.button_frame, text='< Back', command=lambda: self.change_page('back'))
        self.back_button.place(x=width - 160, y=0)
        self.next_button = tk.Button(self.button_frame, text='Next >', command=lambda: self.change_page('next'))
        self.next_button.place(x=width - 80, y=0)

        self.change_page()

    def change_page(self, direction: str = None):
        step = self.current_step if self.current_step is not None else 0

        if direction == 'next':
            current_step = self.steps[self.current_step]
            if current_step.validate():
                self.parameters = {**self.parameters, **current_step.get_values()}

                if self.current_step == len(self.steps) - 1:
                    self.finish()
                    return

                for step in range(self.current_step+1, len(self.steps)):
                    if self.steps[step].display():
                        break
            else:
                print('Failed to validate')

        elif direction == 'back':
            for step in reversed(range(0, self.current_step)):
                if self.steps[step].display():
                    break

        if self.current_step is not None:  # remove current step
            self.steps[self.current_step].grid_remove()

        self.current_step = step
        self.steps[step].grid(column=0, row=0)

        if step == 0:
            self.back_button.config(state='disabled')
            self.next_button.config(state='normal', text='Next >')
            self.cancel_button.config(state='normal')

        elif step == len(self.steps)-1:
            self.back_button.config(state='normal')
            self.next_button.config(state='normal', text='Finish')
            self.cancel_button.config(state='normal')

        else:
            self.back_button.config(state='normal')
            self.next_button.config(state='normal', text='Next >')
            self.cancel_button.config(state='normal')

    def cancel(self):
        title = 'Exit Setup'
        message = 'Are you sure you want to exit the setup?'
        if messagebox.askquestion(title, message, icon='warning') == 'yes':
            self.quit()

    def finish(self):
        print(self.parameters)
        self.quit()


if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(width=False, height=False)
    Wizard(root).pack(side='left', fill='both', expand=True)
    root.mainloop()
