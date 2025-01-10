classdef guiabout_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure         matlab.ui.Figure
        GridLayout       matlab.ui.container.GridLayout
        LeftPanel        matlab.ui.container.Panel
        Label_4          matlab.ui.control.Label
        Label_3          matlab.ui.control.Label
        VTILabel         matlab.ui.control.Label
        V01Label         matlab.ui.control.Label
        Label_2          matlab.ui.control.Label
        StarrMoonnLabel  matlab.ui.control.Label
        RightPanel       matlab.ui.container.Panel
        Image            matlab.ui.control.Image
    end

    % Properties that correspond to apps with auto-reflow
    properties (Access = private)
        onePanelWidth = 576;
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Changes arrangement of the app based on UIFigure width
        function updateAppLayout(app, event)
            currentFigureWidth = app.UIFigure.Position(3);
            if(currentFigureWidth <= app.onePanelWidth)
                % Change to a 2x1 grid
                app.GridLayout.RowHeight = {330, 330};
                app.GridLayout.ColumnWidth = {'1x'};
                app.RightPanel.Layout.Row = 2;
                app.RightPanel.Layout.Column = 1;
            else
                % Change to a 1x2 grid
                app.GridLayout.RowHeight = {'1x'};
                app.GridLayout.ColumnWidth = {218, '1x'};
                app.RightPanel.Layout.Row = 1;
                app.RightPanel.Layout.Column = 2;
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Get the file path for locating images
            pathToMLAPP = fileparts(mfilename('fullpath'));

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.AutoResizeChildren = 'off';
            app.UIFigure.Position = [100 100 685 330];
            app.UIFigure.Name = 'MATLAB App';
            app.UIFigure.SizeChangedFcn = createCallbackFcn(app, @updateAppLayout, true);

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {218, '1x'};
            app.GridLayout.RowHeight = {'1x'};
            app.GridLayout.ColumnSpacing = 0;
            app.GridLayout.RowSpacing = 0;
            app.GridLayout.Padding = [0 0 0 0];
            app.GridLayout.Scrollable = 'on';

            % Create LeftPanel
            app.LeftPanel = uipanel(app.GridLayout);
            app.LeftPanel.Layout.Row = 1;
            app.LeftPanel.Layout.Column = 1;

            % Create StarrMoonnLabel
            app.StarrMoonnLabel = uilabel(app.LeftPanel);
            app.StarrMoonnLabel.Position = [55 204 104 33];
            app.StarrMoonnLabel.Text = '作者：StarrMoonn';

            % Create Label_2
            app.Label_2 = uilabel(app.LeftPanel);
            app.Label_2.Position = [54 162 101 33];
            app.Label_2.Text = '日期：2025.01.10';

            % Create V01Label
            app.V01Label = uilabel(app.LeftPanel);
            app.V01Label.Position = [55 118 100 33];
            app.V01Label.Text = '版本：V0.1';

            % Create VTILabel
            app.VTILabel = uilabel(app.LeftPanel);
            app.VTILabel.Position = [54 253 108 22];
            app.VTILabel.Text = 'VTI介质全波形反演';

            % Create Label_3
            app.Label_3 = uilabel(app.LeftPanel);
            app.Label_3.Position = [47 72 121 22];
            app.Label_3.Text = '最 是 人 间 留 不 住，';

            % Create Label_4
            app.Label_4 = uilabel(app.LeftPanel);
            app.Label_4.Position = [47 40 121 22];
            app.Label_4.Text = '朱 颜 辞 镜 花 辞 树！';

            % Create RightPanel
            app.RightPanel = uipanel(app.GridLayout);
            app.RightPanel.Layout.Row = 1;
            app.RightPanel.Layout.Column = 2;

            % Create Image
            app.Image = uiimage(app.RightPanel);
            app.Image.Position = [15 19 452 293];
            app.Image.ImageSource = fullfile(pathToMLAPP, 'assets', 'logo_zhanqiao.jpg');

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = guiabout_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end